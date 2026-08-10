# frozen-ruler

**A benchmark is only a ruler if it cannot move.**

A retrieval evaluation harness for markdown knowledge bases — hybrid search
(dense embeddings + BM25 via reciprocal rank fusion, optional cross-encoder
rerank and a PageRank graph lane) with the evaluation discipline as the actual
product: frozen benchmark sets, drift refusal, per-class scoring, honest
statistics for small n.

Extracted from a production system I run daily over my own business's knowledge
base. The example corpus here is synthetic (a tiny espresso wiki); the design
decisions all come from real incidents, several of which are documented below.

## Why "frozen"

Most retrieval demos re-generate their eval set whenever it's convenient, which
makes every number incomparable with the last one. This harness treats the
benchmark like an instrument:

- **The set is sha256-pinned** (`freeze.py` writes a manifest). `score.py`
  **fatal-exits** if the set no longer matches its pin — a modified ruler
  invalidates every stored comparison, so refusing to run is the correct
  behavior. Need to change the set? Cut a new versioned file.
- **Index provenance is stamped into every result** (model, chunk count, build
  time). Without this, a before/after comparison silently attributes *corpus
  drift* to the change under test. This is not hypothetical: a metadata
  writeback once re-embedded the corpus between two runs and moved every
  number; the drift warning in `--compare` exists because of that day.
- **Fully stale gold is reported loudly.** If ALL of an item's gold documents
  disappear from the index, the item is excluded and named in the output. An
  item whose gold set is only *partially* present is scored against the
  surviving subset — a documented trade-off that keeps long-lived sets usable
  across corpus churn (see the docstring in `score.py`).

## What it measures

- recall@k, F1@k (k = 1, 3, 5, 10), MRR@10 — at document level.
- **Per-class breakout** — the set carries a `category` per item (derived,
  synonym, cross-page, partial-recall, negative). A macro-average happily hides
  one class sitting at zero while the rest look fine; the per-class table is
  the whole point.
- **Negatives and abstention** — negative items (no correct answer in the
  corpus) are scored by a different rule, and the scorer reports the cosine
  separation AUC between positives and negatives: the probability that a random
  positive query's top hit outscores a random negative's. That number tells you
  whether "return nothing rather than a bad answer" is even learnable from your
  retrieval scores.
- **Mega-document contamination** — the share of top-5 slots taken by the N
  largest documents when they aren't gold. Big append-style documents crowd out
  everything else; this makes the crowding visible.

## Honest statistics for small n

`arm_compare.py` compares two scored runs over the same frozen set with a
paired sign test (two-sided binomial), Wilson 95% CIs, and the discordant-pair
counts that drive the test — because at benchmark sizes like n=200, the honest
headline is usually "indistinguishable," and the tooling should make that easy
to say. It also prints which items flipped at recall@1 (up to six per direction),
so a significant delta can be read, not just believed.

## Results from production use

Numbers below are from the private benchmark this was extracted from — a
59-item v1 frozen set, later re-cut as a 209-item v2 (189 scored positives +
negatives) with per-class structure. Not reproducible from the synthetic
example; the example corpus is six notes and scores are trivially high on it.

- **Adopted** bge-base over bge-small on evidence (v1 set, n=59): recall@5
  0.449 → 0.576, paired 9-improved / 0-regressed, p = 0.004.
- **Rejected** bge-large twice (v2 set): no metric reached significance (min p = 0.34)
  and the synonym class *regressed* (recall@1 0.400 → 0.360). Documented so the
  question doesn't get re-litigated without a new model family.
- **Killed my own shipped feature** (v2 set): a graph-retrieval lane I had
  already shipped scored 0-better / 9-worse at recall@10, p = 0.004.
  Demoted it; kept the original favorable numbers under a HISTORICAL heading
  instead of deleting them.

The harness exists so that decisions like these are forced by the numbers,
including when the numbers say your own idea was wrong.

## Quickstart

Requires Python 3.10+; dependencies are pulled per-run with
[uv](https://github.com/astral-sh/uv) (no environment to manage):

```bash
# 1. Build an index over the example corpus
CORPUS_ROOT=$PWD/example/corpus VSS_INDEX_DIR=$PWD/example/index \
  uv run --isolated --with "sentence-transformers==5.6.0" --with "numpy==2.4.6" \
  python3 build_index.py

# 2. Score the frozen example benchmark
CORPUS_ROOT=$PWD/example/corpus VSS_INDEX_DIR=$PWD/example/index \
  uv run --isolated --with "sentence-transformers==5.6.0" --with "numpy==2.4.6" \
  python3 score.py --set example/benchmark.jsonl --out baseline.json

# 3. Change something (e.g. VSS_MODEL=BAAI/bge-small-en-v1.5), rescore to a
#    second file, then compare the two arms
CORPUS_ROOT=$PWD/example/corpus VSS_INDEX_DIR=$PWD/example/index2 \
  VSS_MODEL=BAAI/bge-small-en-v1.5 \
  uv run --isolated --with "sentence-transformers==5.6.0" --with "numpy==2.4.6" \
  python3 build_index.py
CORPUS_ROOT=$PWD/example/corpus VSS_INDEX_DIR=$PWD/example/index2 \
  VSS_MODEL=BAAI/bge-small-en-v1.5 \
  uv run --isolated --with "sentence-transformers==5.6.0" --with "numpy==2.4.6" \
  python3 score.py --set example/benchmark.jsonl --out candidate.json
uv run python3 arm_compare.py baseline.json candidate.json base cand

# 4. Author your own set, then freeze it
uv run python3 freeze.py path/to/benchmark.jsonl

# Interactive search over the index
CORPUS_ROOT=$PWD/example/corpus VSS_INDEX_DIR=$PWD/example/index \
  uv run --isolated --with "sentence-transformers==5.6.0" --with "numpy==2.4.6" \
  python3 search.py "why do my shots run fast" --k 5
```

Point `CORPUS_ROOT` at any directory of markdown files to use your own corpus.

## Benchmark set format

One JSON object per line:

```json
{"id": "syn001", "category": "synonym", "query": "frothing technique for silky foam", "gold_paths": ["milk steaming basics.md"]}
{"id": "neg001", "category": "negative", "query": "how do I roast green beans", "gold_paths": [], "acceptable_paths": []}
```

`gold_paths` are corpus-relative. Negative items have empty `gold_paths`;
retrieval always returns k candidates, so a negative is judged by whether what
ranks on top is in `acceptable_paths`, and true abstention is a downstream
call built on the reported cosine separation.

## Design decisions worth stealing

- **Atomic index publish** — data files first, manifest last as the commit
  marker; a torn write surfaces as a count mismatch at load instead of silent
  wrong results.
- **Collapse guard** — once the index holds 50+ documents, the builder refuses
  to run if the corpus count drops >50% (cloud-storage eviction once made
  "publish an empty index with exit 0" a real failure mode).
- **Carry-forward on read failure** — a transiently unreadable file keeps its
  old vectors rather than silently vanishing from the index; mass read
  failures with no prior vectors to carry forward abort the build entirely.
- **Query-only instruction prefix** — the bge retrieval instruction is
  prepended to queries, never to passages.
- **Contextual chunk prefixes** — each indexed chunk carries a synthetic
  prefix (document title + opening sentence + heading path) on the *indexed*
  form; the stored display form keeps only the title/heading context.

## Tests

`uv run python3 test_stats.py` — exact checks on the hand-rolled statistics
(sign-test binomial p-values against closed-form values, Wilson interval
against reference constants, and the empty/degenerate guards).

## What this is not

Not a library, not a package, no API stability promise. It's a reference
implementation of an evaluation discipline: pin the ruler, stamp the
provenance, break out the classes, and let paired statistics tell you when
your improvement is actually noise.

## License

MIT
