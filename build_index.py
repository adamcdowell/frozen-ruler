"""Build / incrementally refresh the dense embedding index over the corpus.

Run (registry install, clean tier, isolated env):
  uv run --isolated --with "sentence-transformers==5.6.0" --with "numpy==2.4.6" \
    python build_index.py

Safe for the unattended weekly loop:
  - exclusive non-blocking lock (overlapping runs skip, exit 3)
  - refuses to run if the corpus is missing or its note count collapsed >50%
    (so cloud-storage eviction / unmount can't wipe a good index)
  - atomic publish (temp files + os.replace, manifest written LAST as the commit
    marker) so a mid-write kill can never leave a torn index
  - incremental: unchanged notes reuse vectors; changed re-embed; deleted drop;
    a transient read error CARRIES THE NOTE FORWARD rather than dropping it
"""
import os, sys, json, time, fcntl, numpy as np
import common as C
from sentence_transformers import SentenceTransformer


def load_existing():
    """(index_or_None, prev_notes_hint). The hint survives even when the index is
    discarded for a model/chunker change, so the 50%-collapse guard stays armed
    during full rebuilds — otherwise a mass read failure (cloud-storage eviction)
    publishes a gutted index with exit 0."""
    try:
        man, chunks, emb = C.load_index()
    except FileNotFoundError:
        return None, 0
    except Exception as e:
        print(f"[build] existing index unreadable/corrupt ({e}); full rebuild", file=sys.stderr)
        return None, 0
    hint = man.get("notes", 0)
    if man.get("model") != C.MODEL:
        print("[build] model changed -> full rebuild", file=sys.stderr)
        return None, hint
    if man.get("chunker", 1) != C.CHUNKER_VERSION:
        print(f"[build] chunker v{man.get('chunker', 1)} -> v{C.CHUNKER_VERSION}: "
              f"full re-embed (per-file hashes can't see chunker changes)", file=sys.stderr)
        return None, hint
    return (man, chunks, emb), hint


def atomic_publish(emb, meta, manifest):
    d = C.INDEX_DIR
    tmp = {"emb": ".embeddings.npy.tmp", "ch": ".chunks.json.tmp", "man": ".manifest.json.tmp"}
    with open(os.path.join(d, tmp["emb"]), "wb") as f:
        np.save(f, emb); f.flush(); os.fsync(f.fileno())
    with open(os.path.join(d, tmp["ch"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    with open(os.path.join(d, tmp["man"]), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    # data first, manifest LAST = commit marker. A kill between replaces leaves a
    # count mismatch that load_index() detects, never silent wrong results.
    os.replace(os.path.join(d, tmp["emb"]), os.path.join(d, "embeddings.npy"))
    os.replace(os.path.join(d, tmp["ch"]), os.path.join(d, "chunks.json"))
    os.replace(os.path.join(d, tmp["man"]), os.path.join(d, "manifest.json"))


def run():
    if not os.path.isdir(C.CORPUS):
        sys.exit(f"[build] FATAL: corpus not found at {C.CORPUS}; index left untouched")
    cur = {C.rel(f): f for f in C.walk()}
    existing, prev_notes = load_existing()
    if prev_notes >= 50 and len(cur) < prev_notes * 0.5:
        sys.exit(f"[build] FATAL: note count collapsed ({len(cur)} < 50% of {prev_notes}); "
                 f"aborting, index left untouched")
    if not cur:
        sys.exit("[build] FATAL: no markdown notes found; aborting, index left untouched")

    old_files = existing[0]["files"] if existing else {}
    dim = existing[2].shape[1] if existing else None
    old_rows = {}
    if existing:
        _, ochunks, oemb = existing
        for i, m in enumerate(ochunks):
            old_rows.setdefault(m["path"], []).append((m, oemb[i]))

    reuse_meta, reuse_vecs, to_embed, to_embed_meta, newman = [], [], [], [], {}
    reused = changed = skipped = 0
    for rp, full in sorted(cur.items()):
        h, chs = C.chunk_note(full)
        if h is None:  # read failure: carry forward old vectors if we have them
            if rp in old_rows:
                for m, v in old_rows[rp]:
                    reuse_meta.append(m); reuse_vecs.append(v)
                newman[rp] = old_files.get(rp, "")
                reused += 1
            else:
                skipped += 1
            continue
        newman[rp] = h
        if old_files.get(rp) == h and rp in old_rows:
            for m, v in old_rows[rp]:
                reuse_meta.append(m); reuse_vecs.append(v)
            reused += 1
        else:
            changed += 1
            for ci, (txt, ctxt) in enumerate(chs):
                # ctext (synthetic contextual prefix, chunker v4) is what gets
                # embedded + BM25-indexed; text stays the raw display/preview form
                to_embed.append(ctxt)
                to_embed_meta.append({"path": rp, "chunk": ci, "text": txt, "ctext": ctxt})

    # read failures ABORT rather than shrink the index: with no carry-forward rows
    # (full rebuild) a mass open() failure would otherwise publish a gutted index
    if skipped > max(5, len(cur) // 10):
        sys.exit(f"[build] FATAL: {skipped} of {len(cur)} notes unreadable — "
                 f"aborting, index left untouched")
    if existing and changed == 0 and skipped == 0 and set(newman) == set(old_files):
        print("[build] no content changes — index untouched (skipping republish)",
              file=sys.stderr)
        return

    model = new_vecs = None
    if to_embed:
        model = SentenceTransformer(C.MODEL)
        new_vecs = model.encode(to_embed, normalize_embeddings=True,
                                batch_size=64, show_progress_bar=False).astype("float32")
        dim = new_vecs.shape[1]
    if dim is None:  # nothing to embed and no prior index — ask the model
        model = model or SentenceTransformer(C.MODEL)
        dim = model.get_sentence_embedding_dimension()

    parts = []
    if reuse_vecs:
        parts.append(np.asarray(reuse_vecs, dtype="float32"))
    if new_vecs is not None and len(new_vecs):
        parts.append(new_vecs)
    emb = np.vstack(parts) if parts else np.zeros((0, dim), dtype="float32")
    meta = reuse_meta + to_embed_meta
    if emb.shape[0] != len(meta):
        sys.exit(f"[build] FATAL: internal emb/meta mismatch ({emb.shape[0]} vs {len(meta)})")

    manifest = {"model": C.MODEL, "chunker": C.CHUNKER_VERSION,
                "dim": int(emb.shape[1]), "count": int(emb.shape[0]),
                "notes": len(newman), "built_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "files": newman}
    atomic_publish(emb, meta, manifest)
    print(f"[build] indexed notes={len(newman)} (reused {reused}, re-embedded {changed}, "
          f"unreadable-skipped {skipped}) chunks={emb.shape[0]} dim={emb.shape[1]}", file=sys.stderr)


def main():
    os.makedirs(C.INDEX_DIR, exist_ok=True)
    lockfd = open(os.path.join(C.INDEX_DIR, ".build.lock"), "w")
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # exit 3, not 0: callers must not mistake a lock-skip for a fresh index
        print("[build] another build is in progress; skipping", file=sys.stderr)
        sys.exit(3)
    try:
        run()
    finally:
        try:
            fcntl.flock(lockfd, fcntl.LOCK_UN)
        finally:
            lockfd.close()


if __name__ == "__main__":
    main()
