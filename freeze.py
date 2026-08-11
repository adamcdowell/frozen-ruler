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
    # A ruler with two marks in the same place is not a ruler. Duplicate ids make
    # score.py's aggregate (a list, every row counted) and arm_compare.py's paired
    # sample (a dict keyed by id, duplicates collapsed) describe different n. A gold
    # path repeated inside one item caps that item's recall below 1.0, because the
    # hit count is a set intersection and the denominator is not.
    no_id = [n for n, it in enumerate(items, 1) if not it.get("id")]
    dup_ids = sorted(i for i, n in Counter(it.get("id") for it in items).items() if i and n > 1)
    dup_gold = sorted(it.get("id", "?") for it in items
                      if len(it.get("gold_paths", [])) != len(set(it.get("gold_paths", []))))
    if no_id or dup_ids or dup_gold:
        sys.exit(f"REFUSING to freeze {os.path.basename(path)}: "
                 + (f"lines missing an id: {no_id}. " if no_id else "")
                 + (f"duplicate ids: {dup_ids}. " if dup_ids else "")
                 + (f"items repeating a gold_path: {dup_gold}. " if dup_gold else "")
                 + "Fix the set before freezing — a frozen set is the ruler.")

    # SCHEMA GATE: in score.py the `category` LABEL is the authority — an item is
    # a negative iff category == "negative", not because gold_paths is empty. So
    # refuse to freeze a set where label and gold set disagree: otherwise a
    # mislabelled negative is either misreported as STALE (empty gold, no label)
    # or scored as a negative with its gold silently ignored (label, real gold).
    bad = []
    for i, it in enumerate(items, 1):
        if "id" not in it or "query" not in it:
            bad.append(f"line {i}: missing id/query")
            continue
        cat, gold = it.get("category"), it.get("gold_paths")
        if gold is None:
            bad.append(f"{it['id']}: no gold_paths key")
        elif cat == "negative" and gold:
            bad.append(f"{it['id']}: category=negative but gold_paths is non-empty")
        elif cat != "negative" and gold == []:
            bad.append(f"{it['id']}: empty gold_paths but category={cat!r} — a "
                       f"negative must be labelled category=\"negative\"")
    if bad:
        sys.exit("REFUSING TO FREEZE — {} malformed item(s):\n  {}".format(
            len(bad), "\n  ".join(bad)))

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
