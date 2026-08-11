"""Shared config + helpers: corpus walking, chunking, index loading.

Point CORPUS_ROOT at any directory of markdown files. Index location defaults
to ./index beside these scripts (override with VSS_INDEX_DIR).
"""
import os, re, sys, hashlib, json
import numpy as np

CORPUS = os.environ.get("CORPUS_ROOT", os.path.join(os.getcwd(), "example", "corpus")).rstrip(os.sep)
# Pruned by directory NAME (segment match, not substring) so a stray substring
# can't over-exclude and a deep folder can't slip past.
EXCLUDE_DIRS = {".obsidian", "attachments", ".git", "Templates", ".trash"}
# VSS_MODEL / VSS_INDEX_DIR env overrides exist for A/B experiments (build a
# candidate index elsewhere, score it with score.py) — defaults stay stable so
# everyday search never changes behavior mid-experiment.
MODEL = os.environ.get("VSS_MODEL", "BAAI/bge-base-en-v1.5")
# bge retrieval instruction — prepended to QUERIES only, never to passages.
QPREFIX = "Represent this sentence for searching relevant passages: "

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.environ.get("VSS_INDEX_DIR", os.path.join(HERE, "index"))
CHUNK_CHARS = 1200   # target chunk size (chars, measured pre-collapse)
CHUNK_MIN = 200      # a tail piece smaller than this merges backward
MERGE_MAX = 1400     # ctx + merged piece ceiling — keeps chunks inside the BGE
                     # 512-token window (1800-char merges measured to overflow it)
MAX_CHUNKS = 256     # hard stop for pathological files — LOGGED, never silent
                     # (an early version silently truncated large files)
TAIL_KEEP = 64       # over-cap files keep head + this many TAIL chunks — append-style
                     # logs grow at the end, so head-only truncation would drop
                     # exactly the newest entries
CHUNKER_VERSION = 4  # bump forces a full re-embed on the next build.
                     # v4: synthetic contextual prefix on the INDEXED text only
                     # (title + note-opening sentence); display text unchanged.
OPENING_CAP = 300    # first-sentence cap (chars) for the synthetic prefix.
                     # Worst-case ctext ~= MERGE_MAX + ~355 boilerplate: only the
                     # largest merged chunks graze the 512-token window, and only
                     # the dense side truncates the tail (BM25 sees full ctext).


def walk(root=CORPUS):
    """Markdown files under root. Prunes excluded dirs by name, never follows
    symlinks, and rejects any file whose real path escapes the corpus (so a stray
    symlink can't widen the read scope beyond the corpus)."""
    root_real = os.path.realpath(root)
    out = []
    for dp, dn, fn in os.walk(root, followlinks=False):
        dn[:] = [d for d in dn if d not in EXCLUDE_DIRS]
        for f in fn:
            if not f.endswith(".md"):
                continue
            p = os.path.join(dp, f)
            rp = os.path.realpath(p)
            if rp == root_real or rp.startswith(root_real + os.sep):
                out.append(p)
    return out


def rel(p, root=CORPUS):
    return p[len(root) + 1:]


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
_FENCE = re.compile(r"^(```|~~~)", re.M)


def _fence_spans(body):
    """[(start, end)] spans of fenced code blocks; an unclosed fence runs to EOF."""
    spans, open_at = [], None
    for m in _FENCE.finditer(body):
        if open_at is None:
            open_at = m.start()
        else:
            spans.append((open_at, m.end()))
            open_at = None
    if open_at is not None:
        spans.append((open_at, len(body)))
    return spans


def _sections(body):
    """[(heading_path, text)] — split on ATX headings, tracking the trail
    (h1 > h2 > ...) so every chunk can carry where-in-the-note context.
    Fenced code blocks are masked ('#' comment lines are not headings), and a
    heading that owns no body text still gets emitted as its own tiny section so
    its words reach the index (headings-only outlines were unsearchable)."""
    fences = _fence_spans(body)
    matches = [m for m in _HEADING.finditer(body)
               if not any(s <= m.start() < e for s, e in fences)]
    secs, trail = [], []  # trail entries: [lvl, text, emitted?]

    def emit(path_str, start, end):
        txt = body[start:end]
        if txt.strip():
            secs.append((path_str, txt))
            return True
        return False

    def flush(entry):
        """A trail entry leaving scope without ever being emitted: index its words."""
        if not entry[2]:
            secs.append((entry[1], entry[1]))

    if not matches:
        emit("", 0, len(body))
        return secs
    if matches[0].start() > 0:
        emit("", 0, matches[0].start())
    for i, m in enumerate(matches):
        lvl = len(m.group(1))
        while trail and trail[-1][0] >= lvl:
            flush(trail.pop())
        trail.append([lvl, m.group(2), False])
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        if emit(" > ".join(t for _, t, _e in trail), m.end(), end):
            for entry in trail:
                entry[2] = True
    for entry in trail:
        flush(entry)
    return secs


_FRONTMATTER = re.compile(r"\A---[ \t]*\n.*?\n(?:---|\.\.\.)[ \t]*\n", re.S)


def _first_sentence(body, cap=None):
    """First sentence of the note body for the synthetic contextual prefix: frontmatter
    stripped, fenced code masked, heading lines dropped (the title already names
    the note), list/quote markers removed. Capped at OPENING_CAP chars; empty
    string when the note has no prose."""
    cap = OPENING_CAP if cap is None else cap
    text = _FRONTMATTER.sub("", body, count=1)
    for s, e in reversed(_fence_spans(text)):
        text = text[:s] + text[e:]
    lines = []
    for ln in text.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", ln):
            continue  # headings echo the title/outline, not an opening sentence
        lines.append(re.sub(r"^\s{0,3}(?:>|[-*+]|\d+[.)])\s+", "", ln))
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not text:
        return ""
    m = re.search(r"[.!?](?=\s|$)", text)
    sent = text[:m.end()] if m else text
    return sent[:cap].strip()


def _split_point(s, target):
    """Best cut <= target: sentence end, else word boundary, else hard cut."""
    window = s[:target]
    last = None
    for m in re.finditer(r"[.!?]\s", window):
        last = m
    if last and last.end() > target * 0.5:
        return last.end()
    sp = window.rfind(" ")
    if sp > target * 0.5:
        return sp + 1
    return target


def _pack(pieces, target):
    """Greedy paragraph packing to ~target chars; oversized single pieces get
    sentence/word-boundary splits so no content is ever dropped."""
    assert target >= 1, "chunk target must be positive"
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) + 1 > target:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}" if buf else p
        while len(buf) > target:
            cut = _split_point(buf, target) or len(buf)  # never 0: guards a hang
            out.append(buf[:cut].strip())
            buf = buf[cut:].strip()
    if buf:
        out.append(buf)
    return out


def chunk_note(path):
    """(content_hash, [(text, ctext), ...]) on success; (None, None) on a genuine
    READ FAILURE (file locked / gone / permission), so the caller can carry the
    note forward instead of silently dropping it. Encoding glitches are tolerated
    (errors='replace'), not treated as failures.

    Heading-aware recursive chunking (sections -> paragraphs -> sentence splits),
    each chunk prefixed with "title — heading path" context. The hash is a raw
    body SHA1; CHUNKER_VERSION forces re-embeds on chunker changes.

    Contextual retrieval: each chunk yields a pair —
    `text` (the v3 display form, "title — heading path. piece", UNCHANGED) and
    `ctext` (the INDEXED form fed to the embedder and BM25): a synthetic prefix
    'This passage is from the note "<title>". The note opens: <first sentence>'
    followed by the heading path (title dropped there — the prefix already names
    it, so the net is title + opening + heading path + chunk) and the piece."""
    title = os.path.splitext(os.path.basename(path))[0]
    try:
        body = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return None, None
    h = hashlib.sha1(body.encode("utf-8", "ignore")).hexdigest()

    chunks = []  # (context, piece)
    for hpath, txt in _sections(body):
        ctx = f"{title} — {hpath}" if hpath else title
        paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        for piece in _pack(paras, CHUNK_CHARS):
            chunks.append((ctx, piece))

    merged = []  # fold tiny tails back into their section neighbor
    for ctx, piece in chunks:
        if (merged and len(piece) < CHUNK_MIN and merged[-1][0] == ctx
                and len(ctx) + len(merged[-1][1]) + len(piece) < MERGE_MAX):
            merged[-1] = (ctx, f"{merged[-1][1]} {piece}")
        else:
            merged.append((ctx, piece))

    if len(merged) > MAX_CHUNKS:
        print(f"[chunk] {rel(path)} truncated: {len(merged)} chunks -> {MAX_CHUNKS} cap "
              f"(kept head {MAX_CHUNKS - TAIL_KEEP} + tail {TAIL_KEEP})", file=sys.stderr)
        merged = merged[:MAX_CHUNKS - TAIL_KEEP] + merged[-TAIL_KEEP:]
    if not merged:
        merged = [(title, "")]

    prefix = f'This passage is from the note "{title}".'
    opening = _first_sentence(body)
    if opening:
        prefix = f"{prefix} The note opens: {opening}"
    out = []
    for ctx, piece in merged:
        text = re.sub(r"\s+", " ", f"{ctx}. {piece}").strip()
        # ctx is title or "title — hpath"; keep only hpath for ctext (the
        # synthetic prefix already names the title — no double-title)
        hpath = ctx[len(title) + 3:] if ctx.startswith(f"{title} — ") else \
            ("" if ctx == title else ctx)
        tail = f"{hpath}. {piece}" if hpath else piece
        ctext = re.sub(r"\s+", " ", f"{prefix} {tail}").strip()
        out.append((text, ctext))
    return h, out


def load_index():
    """Load + validate the index. Raises FileNotFoundError if absent, ValueError
    if the manifest and the data files it NAMES disagree on row count.

    manifest.json is the single commit marker: it names its generation of data
    files (`emb` / `chunks`), which are written under fresh names and never
    overwrite the live pair, so a publish is one atomic rename and a reader gets
    the whole old snapshot or the whole new one. The built_iso re-read below
    covers the remaining race: a manifest replaced mid-read, whose generation
    may since have been reclaimed. An index built before generation-stamped
    publishes still loads via the flat-name fallback — but it predates this
    guarantee; rebuild it."""
    man_path = os.path.join(INDEX_DIR, "manifest.json")
    for attempt in (1, 2):
        man = json.load(open(man_path, encoding="utf-8"))
        emb_name = man.get("emb", "embeddings.npy")       # legacy flat layout
        ch_name = man.get("chunks", "chunks.json")
        try:
            with open(os.path.join(INDEX_DIR, emb_name), "rb") as f:
                emb = np.load(f, allow_pickle=False)  # never unpickle index data
            chunks = json.load(open(os.path.join(INDEX_DIR, ch_name), encoding="utf-8"))
        except FileNotFoundError:
            # our generation was reclaimed by a publish that landed mid-read
            if attempt == 1:
                continue
            raise ValueError("index is being republished; retry the query")
        man2 = json.load(open(man_path, encoding="utf-8"))
        if man.get("built_iso") != man2.get("built_iso"):
            if attempt == 1:
                continue  # a publish landed mid-load; one clean retry
            raise ValueError("index is being republished; retry the query")
        if not (man.get("count") == emb.shape[0] == len(chunks)):
            raise ValueError(f"index inconsistent (manifest.count={man.get('count')} "
                             f"emb={emb.shape[0]} chunks={len(chunks)}); rebuild with build_index.py")
        return man, chunks, emb
