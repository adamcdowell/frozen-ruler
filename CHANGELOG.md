# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/).
This project does not cut versioned releases yet; changes accumulate under Unreleased.
Version numbers, when cut, will follow PEP 440 (matching `pyproject.toml`).

## [Unreleased]

### Added

- Continuous integration (`.github/workflows/ci.yml`): the statistics tests (`test_stats.py`) now run on every push across Python 3.11–3.14 on Linux, macOS and Windows. The lane installs nothing — the statistics are stdlib-only, so they are verifiable without the ~2 GB of torch the indexing side pulls.
- Packaging metadata (`pyproject.toml`) with the dependencies pinned, so every quickstart command drops its `--isolated --with "sentence-transformers==..." --with "numpy==..."` prefix. `pip install -e .` works as the non-uv path. `uv.lock` is committed too — a harness whose argument is that the ruler must not move should pin its own transitive tree.
- An "Example output" section in the README showing real `score.py` and `arm_compare.py` output from the example corpus, including a worked case where recall@1 falls 0.22 and the sign test still returns p=0.5000.
- A Mermaid pipeline diagram of corpus → index → frozen set → score → compare.
- This changelog.

### Fixed

- **"Atomic publish" was not atomic across files.** `atomic_publish()` replaced embeddings, chunks and manifest in three separate `os.replace` calls, and `load_index()` validated only that the three row counts agreed. A build killed between the embeddings and chunks renames left new vectors against stale metadata; because a republish only happens when something changed, and any edit that doesn't cross a chunk boundary keeps the count identical, the equal-count case is the normal one — and since changed files are appended to the tail of the array, a one-word edit to a single note mispaired 6 of 9 rows in the example index while every load check passed. The manifest is now a true commit marker: data files are written under generation-stamped names and published by a single `os.replace` of `manifest.json` (plus directory fsyncs, so the renames can't be reordered by a power loss). Superseded generations are retained one cycle, then reclaimed.
- **`freeze.py` now refuses a set with duplicate ids or a gold path repeated inside one item** — the aggregate counted every row while the paired sign test could only use one row per id, and a repeated gold path capped that item's recall below 1.0. No published result changes: both errors under-report.
- **Documented Python floor was wrong.** The README claimed "Requires Python 3.10+", but the pinned `numpy==2.4.6` requires >=3.11, so a 3.10 environment could never have satisfied the dependencies. Corrected to 3.11+ in the README and declared as `requires-python = ">=3.11"`. Surfaced by uv's resolver the moment `pyproject.toml` was added. (The stdlib-only statistics lane does still run on 3.10.)

[Unreleased]: https://github.com/adamcdowell/frozen-ruler/commits/main
