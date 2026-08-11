# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/).
This project does not cut versioned releases yet; changes accumulate under Unreleased.
Version numbers, when cut, will follow PEP 440 (matching `pyproject.toml`).

## [Unreleased]

### Added

- Packaging metadata (`pyproject.toml`) with the dependencies pinned, so every quickstart command drops its `--isolated --with "sentence-transformers==..." --with "numpy==..."` prefix. `pip install -e .` works as the non-uv path. `uv.lock` is committed too — a harness whose argument is that the ruler must not move should pin its own transitive tree.
- An "Example output" section in the README showing real `score.py` and `arm_compare.py` output from the example corpus, including a worked case where recall@1 falls 0.22 and the sign test still returns p=0.5000.
- A Mermaid pipeline diagram of corpus → index → frozen set → score → compare.
- This changelog.

### Fixed

- **Documented Python floor was wrong.** The README claimed "Requires Python 3.10+", but the pinned `numpy==2.4.6` requires >=3.11, so a 3.10 environment could never have satisfied the dependencies. Corrected to 3.11+ in the README and declared as `requires-python = ">=3.11"`. Surfaced by uv's resolver the moment `pyproject.toml` was added. (The stdlib-only statistics lane does still run on 3.10.)

[Unreleased]: https://github.com/adamcdowell/frozen-ruler/commits/main
