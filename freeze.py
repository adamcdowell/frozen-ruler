"""Freeze a benchmark set: pin its sha256 in a manifest so score.py can refuse
to run against a modified ruler.

  uv run python3 freeze.py example/benchmark.jsonl

Writes <setname>-manifest.json beside the set (name derived from the set's
filename, so multiple frozen versions can coexist). A benchmark is only a ruler if
it cannot move: after freezing, never edit the set in place — cut a NEW
versioned file so prior numbers stay comparable.
"""
import sys, os, json, hashlib, time
from collections import Counter


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    raw = open(path, "rb").read()
    items = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    man = {
        "created": time.strftime("%Y-%m-%d"),
        "items": len(items),
        "by_class": dict(Counter(it.get("category", "unclassified") for it in items)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "frozen": True,
        "fence": ("FROZEN — never edit this set. Regenerate as a NEW version so "
                  "prior numbers stay comparable."),
    }
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(os.path.dirname(os.path.abspath(path)), f"{stem}-manifest.json")
    json.dump(man, open(out, "w"), indent=1)
    print(f"frozen: {len(items)} items, sha256 {man['sha256'][:12]}… -> {out}")


if __name__ == "__main__":
    main()
