"""Paired comparison of two scored arms over the SAME frozen benchmark.
Usage: arm_compare.py <baseline.json> <candidate.json> [baseline-label] [candidate-label]

Reports, per metric and per query class: paired sign test (two-sided binomial),
Wilson 95% CI on the any-hit rate, and the discordant-pair count that drives the
test — because at small n the honest headline is usually "indistinguishable".
"""
import json, math, os, sys
from collections import defaultdict

METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"]
CLASS_METRICS = ["recall@1", "recall@5", "recall@10"]

# A run emits 5 headline tests plus 3 per query class. Labelling every one of
# them "significant" at raw p<0.05 is a multiple-comparisons error, so exactly
# one metric is confirmatory and every other row is exploratory, reported with a
# Holm-Bonferroni adjusted p over the whole family emitted by this run.
# Declare the primary BEFORE you look: FROZEN_PRIMARY_METRIC=recall@10 ...
PRIMARY = os.environ.get("FROZEN_PRIMARY_METRIC", "recall@5")


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, returned in input order."""
    m = len(pvals)
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvals[j])):
        running = max(running, min(1.0, pvals[i] * (m - rank)))
        adj[i] = running
    return adj


def verdict(metric, p, p_adj, primary):
    """Confirmatory label for the one pre-declared metric; everything else is
    exploratory and must clear the family-wise adjusted threshold.

    Both labels name the test: the sign test counts how many queries improved vs
    regressed and discards how much each moved, so it speaks to DIRECTION, not to
    the mean delta printed beside it.
    """
    if metric == primary:
        return "  <-- SIGNIFICANT (primary; more queries up than down)" if p < 0.05 else ""
    return "  <-- exploratory, survives Holm (direction)" if p_adj < 0.05 else ""


def sign_test(a, b, metric, ids=None):
    up = down = 0
    for i in (ids if ids is not None else a):
        if i not in b:
            continue
        d = b[i][metric] - a[i][metric]
        if d > 1e-9:
            up += 1
        elif d < -1e-9:
            down += 1
    n = up + down
    if n == 0:
        return up, down, 1.0
    p = min(1.0, sum(math.comb(n, k) for k in range(max(up, down), n + 1)) / 2 ** n * 2)
    return up, down, p


def wilson(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    doc = json.load(open(path))
    rows = {r["id"]: r for r in doc["per_item"]}
    if len(rows) != len(doc["per_item"]):
        # the warning below only compares collapsed to collapsed, so it cannot see this
        sys.exit(f"REFUSING: {path} contains duplicate item ids ({len(doc['per_item'])} "
                 f"rows collapse to {len(rows)}). The aggregate was computed over every "
                 f"row; the paired tests can only use one row per id, so the two would "
                 f"describe different samples. Re-freeze the set and rescore.")
    return doc["aggregate"], rows


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    bpath, cpath = sys.argv[1], sys.argv[2]
    blab = sys.argv[3] if len(sys.argv) > 3 else "baseline"
    clab = sys.argv[4] if len(sys.argv) > 4 else "candidate"
    bagg, brows = load(bpath)
    cagg, crows = load(cpath)
    if bagg.get("benchmark_set") and cagg.get("benchmark_set") and \
            bagg["benchmark_set"] != cagg["benchmark_set"]:
        sys.exit(f"REFUSING: different benchmark sets ({bagg['benchmark_set']} vs "
                 f"{cagg['benchmark_set']}) — these arms are not comparable.")
    shared = [i for i in brows if i in crows]
    if not shared:
        sys.exit("REFUSING: no shared item ids between the two result files.")
    if len(shared) != len(brows) or len(shared) != len(crows):
        print(f"WARNING: item sets differ (baseline {len(brows)}, candidate {len(crows)}, "
              f"shared {len(shared)}); paired stats use the intersection, but the "
              f"aggregate columns below were computed on each arm's full set.")
    print(f"paired items: {len(shared)}   {blab} vs {clab}\n")

    byc = defaultdict(list)
    for i in shared:
        byc[brows[i].get("category", "?")].append(i)
    classes = sorted(byc)

    # Pass 1: run the whole family before printing any of it, so the
    # Holm adjustment knows how many tests this run actually emitted.
    head = [(m,) + sign_test(brows, crows, m, shared) for m in METRICS]
    percls = {c: [(m,) + sign_test(brows, crows, m, byc[c]) for m in CLASS_METRICS]
              for c in classes}
    pvals = [r[3] for r in head] + [r[3] for c in classes for r in percls[c]]
    adj = holm(pvals)
    print(f"FAMILY: {len(pvals)} tests this run ({len(METRICS)} headline + "
          f"{len(CLASS_METRICS)}x{len(classes)} per class). Primary metric: {PRIMARY}. "
          f"Uncorrected p is reported for every row; only the primary is confirmatory, "
          f"the rest are exploratory at Holm-adjusted p.\n")

    print("HEADLINE (all scored items)")
    print(f"  {'metric':10s} {blab:>14s} {clab:>14s}  mean delta   sign test (direction only)")
    for k, (m, up, down, p) in enumerate(head):
        d = cagg[m] - bagg[m]
        star = verdict(m, p, adj[k], PRIMARY)
        # The sign test counts how many queries improved vs regressed and
        # discards how much each moved. It tests DIRECTION, not the mean delta
        # printed to its left, so the two can disagree when a few large moves
        # dominate the mean. Flag the disagreement instead of letting a marker
        # be read as a claim about the average.
        if p < 0.05 and d * (up - down) < 0:
            star += "  [!] DIRECTION AND MEAN DISAGREE — a few large moves dominate the mean"
        print(f"  {m:10s} {bagg[m]:>14.4f} {cagg[m]:>14.4f} {d:>+8.4f}   "
              f"{up:3d}up/{down:3d}dn p={p:.4f} p_holm={adj[k]:.4f}{star}")

    # any-hit rate + Wilson
    print("\nANY-HIT RATE @10 (Wilson 95% CI)")
    for lab, rows in ((blab, brows), (clab, crows)):
        pos = [i for i in shared if rows[i].get("n_gold", 1) > 0]
        if not pos:
            print(f"  {lab:14s}: no positive items — skipping")
            continue
        hits = sum(1 for i in pos if rows[i]["recall@10"] > 0)
        lo, hi = wilson(hits, len(pos))
        print(f"  {lab:14s}: {hits}/{len(pos)} = {hits/len(pos):.3f}   CI [{lo:.3f}, {hi:.3f}]")

    # per class — always exploratory: these slices are not pre-declared
    print("\nPER CLASS (exploratory — hypothesis-generating, not confirmatory)")
    k = len(head)
    for cls in classes:
        ids = byc[cls]
        print(f"\n  [{cls}] n={len(ids)}")
        for m, up, down, p in percls[cls]:
            bm = sum(brows[i][m] for i in ids) / len(ids)
            cm = sum(crows[i][m] for i in ids) / len(ids)
            star = "  <-- survives Holm (direction)" if adj[k] < 0.05 else ""
            if p < 0.05 and (cm - bm) * (up - down) < 0:
                star += "  [!] direction and mean disagree"
            print(f"    {m:10s} {bm:.4f} -> {cm:.4f}  ({cm-bm:+.4f})  "
                  f"{up:2d}up/{down:2d}dn p={p:.3f} p_holm={adj[k]:.3f}{star}")
            k += 1

    # items that flip
    print("\nFLIPS at recall@1 (who actually moved)")
    gained = [i for i in shared if crows[i]["recall@1"] > brows[i]["recall@1"]]
    lost = [i for i in shared if crows[i]["recall@1"] < brows[i]["recall@1"]]
    print(f"  {clab} wins top-1 on {len(gained)}, loses on {len(lost)}")
    for i in gained[:6]:
        print(f"    + [{brows[i].get('category')}] {brows[i]['query'][:66]}")
    for i in lost[:6]:
        print(f"    - [{brows[i].get('category')}] {brows[i]['query'][:66]}")


if __name__ == "__main__":
    main()
