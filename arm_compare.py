"""Paired comparison of two scored arms over the SAME frozen benchmark.
Usage: arm_compare.py <baseline.json> <candidate.json> [baseline-label] [candidate-label]

Reports, per metric and per query class: paired sign test (two-sided binomial),
Wilson 95% CI on the any-hit rate, and the discordant-pair count that drives the
test — because at small n the honest headline is usually "indistinguishable".
"""
import json, math, sys
from collections import defaultdict

METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"]


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

    print("HEADLINE (all scored items)")
    print(f"  {'metric':10s} {blab:>14s} {clab:>14s}   delta     sign test")
    for m in METRICS:
        up, down, p = sign_test(brows, crows, m, shared)
        d = cagg[m] - bagg[m]
        star = "  <-- SIGNIFICANT" if p < 0.05 else ""
        print(f"  {m:10s} {bagg[m]:>14.4f} {cagg[m]:>14.4f} {d:>+8.4f}   {up:3d}up/{down:3d}dn p={p:.4f}{star}")

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

    # per class
    byc = defaultdict(list)
    for i in shared:
        byc[brows[i].get("category", "?")].append(i)
    print("\nPER CLASS (the question v1 could not answer)")
    for cls in sorted(byc):
        ids = byc[cls]
        print(f"\n  [{cls}] n={len(ids)}")
        for m in ("recall@1", "recall@5", "recall@10"):
            bm = sum(brows[i][m] for i in ids) / len(ids)
            cm = sum(crows[i][m] for i in ids) / len(ids)
            up, down, p = sign_test(brows, crows, m, ids)
            star = "  <-- SIGNIFICANT" if p < 0.05 else ""
            print(f"    {m:10s} {bm:.4f} -> {cm:.4f}  ({cm-bm:+.4f})  {up:2d}up/{down:2d}dn p={p:.3f}{star}")

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
