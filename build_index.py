"""Build / incrementally refresh the dense embedding index over the corpus.

Run (registry install, clean tier, isolated env):
  uv run --isolated --with "sentence-transformers==5.6.0" --with "numpy==2.4.6" \
    python build_index.py

Safe for the unattended weekly loop:
  - exclusive non-blocking lock (overlapping runs skip, exit 3)
  - refuses to run if the corpus is missing or its note count collapsed >50%
    (so cloud-storage eviction / unmount can't wipe a good index)
  - atomic publish: generation-stamped data files + ONE os.replace of the
    manifest that names them, so a mid-write kill leaves the previous snapshot
    intact rather than a mixed one
  - incremental: unchanged notes reuse vectors; changed re-embed; deleted drop;
    a transient read error CARRIES THE NOTE FORWARD rather than dropping it
"""
import os, sys, json, time, fcntl, uuid, numpy as np
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


def _fsync_dir(d):
    """Durably commit the directory entry itself: file fsync alone does not
    guarantee a rename survives a power loss, or that renames land in order."""
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_publish(emb, meta, manifest):
    """ONE rename publishes the whole snapshot.

    Data files are written under fresh generation-stamped names, so they never
    overwrite the live pair; manifest.json is then os.replace()d to NAME them.
    A crash anywhere before that single rename leaves the previous manifest
    pointing at its own still-intact files. Three sequential replaces did NOT
    give this: a same-count publish killed between the embeddings and chunks
    renames passed every load check while new vectors sat against stale
    metadata (and the incremental builder moves changed files to the tail, so
    the rows are wholesale reordered, not locally wrong).
    """
    d = C.INDEX_DIR
    gen = uuid.uuid4().hex[:12]
    emb_name, ch_name = f"embeddings-{gen}.npy", f"chunks-{gen}.json"
    with open(os.path.join(d, emb_name), "wb") as f:
        np.save(f, emb); f.flush(); os.fsync(f.fileno())
    with open(os.path.join(d, ch_name), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    try:
        prev = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    except Exception:
        prev = None
    manifest = {**manifest, "emb": emb_name, "chunks": ch_name, "gen": gen}
    tmp = os.path.join(d, ".manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    _fsync_dir(d)                                  # data + tmp durable BEFORE the commit
    os.replace(tmp, os.path.join(d, "manifest.json"))   # <-- the commit, one rename
    _fsync_dir(d)                                  # the commit itself durable
    # Retain the generation we just superseded so a reader already holding the
    # old manifest can still open its files; reclaim everything older.
    keep = {emb_name, ch_name}
    if prev:
        keep |= {prev.get("emb"), prev.get("chunks")}
    for fn in os.listdir(d):
        if (fn.startswith("embeddings-") or fn.startswith("chunks-")) and fn not in keep:
            try:
                os.remove(os.path.join(d, fn))
            except OSError:
                pass


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
