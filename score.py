"""Score a frozen retrieval benchmark against the
LIVE index — the repeatable number for any change to chunking, embeddings, fusion
weights, or reranking. Run BEFORE and AFTER the change; compare.

  uv run --isolated --with "sentence-transformers==5.6.0" --with "numpy==2.4.6" \
    python score.py [--set example/benchmark.jsonl] \
    [--mode hybrid|dense|sparse] [--rerank] [--out results.json] [--compare old-results.json]

Metrics (note-path level, matching what search.py returns): recall@k and f1@k for
k in 1,3,5,10 + MRR@10. Gold paths missing from the index entirely are reported as
STALE and excluded from averages — fix the benchmark item or accept the drift
consciously; never silently. An item whose gold set is only PARTIALLY present is
scored against the present subset (documented choice: it keeps long-lived sets
usable across corpus churn; strict mode = fix the item). Retrieval-only:
gold_answer is not judged here.
"""
import sys, json, argparse
from collections import Counter
import numpy as np
import common as C
import search as S
from sentence_transformers import SentenceTransformer

KS = [1, 3, 5, 10]
MEGA_N = 8        # the N biggest notes by chunk count = the mega-note adversaries
                  # (append-style logs, rollups) whose top-5 crowding gets measured


def ranked_paths(query, mode, rerank, graph, graph_weight, model, chunks, emb,
                 diag=None):
    orders = []
    if mode in ("hybrid", "dense"):
        qv = model.encode([C.QPREFIX + query], normalize_embeddings=True).astype("float32")[0]
        sims = emb @ qv
        if diag is not None:
            # calibrated 0-1 cosine — the ONLY abstention signal available. RRF
            # fused scores are rank-derived (~1/61 for every #1), so they carry no
            # "is this actually relevant" information; negatives must be judged here.
            diag["top_cosine"] = float(sims.max())
        orders.append(list(np.argsort(-sims)))
    if mode in ("hybrid", "sparse"):
        orders.append(list(np.argsort(-S.bm25(query, chunks))))
    fused = S.rrf(orders)
    if rerank:
        fused = S.rerank_pass(query, chunks, fused)
    scores = S.note_scores(fused, chunks)
    if graph:
        g = S.graph_ppr(chunks, scores)
        if g:
            scores = S.fuse_note_lanes(scores, g, graph_weight)
    return sorted(scores, key=lambda p: -scores[p])[: max(KS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None)
    ap.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--graph-weight", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    import os
    bset = a.set or os.path.join(os.path.dirname(os.path.abspath(__file__)), "example", "benchmark.jsonl")

    # FROZEN-SET INTEGRITY: a benchmark is only a ruler if it cannot move. Verify
    # the set against the sha256 pinned in its manifest before scoring anything.
    _stem = os.path.splitext(os.path.basename(bset))[0]
    bman = os.path.join(os.path.dirname(bset), f"{_stem}-manifest.json")
    try:
        pinned = json.load(open(bman)).get("sha256")
        if pinned:
            import hashlib
            actual = hashlib.sha256(open(bset, "rb").read()).hexdigest()
            if actual != pinned:
                sys.exit(f"FATAL: {os.path.basename(bset)} does not match its pinned sha256 "
                         f"({actual[:12]}… vs {pinned[:12]}…). The frozen set was modified — "
                         f"every stored comparison is invalid. Restore it or cut a NEW version.")
            print(f"[frozen-set OK] {os.path.basename(bset)} sha256 {pinned[:12]}… verified")
    except FileNotFoundError:
        print(f"[warn] no manifest beside {os.path.basename(bset)} — integrity unverified",
              file=sys.stderr)

    man, chunks, emb = C.load_index()
    index_paths = {c["path"] for c in chunks}
    model = SentenceTransformer(man.get("model", C.MODEL))
    items = [json.loads(l) for l in open(bset) if l.strip()]

    # v2 items carry `category`; negatives (gold_paths == []) are scored by a
    # different rule and never enter the recall macro-average.
    is_v2 = any("category" in it for it in items)
    mega = {p for p, _ in Counter(c["path"] for c in chunks).most_common(MEGA_N)}

    per_item, stale, negatives = [], [], []
    for it in items:
        cat = it.get("category", "unclassified")
        if is_v2 and cat == "negative":
            diag = {}
            ranked = ranked_paths(it["query"], a.mode, a.rerank, a.graph, a.graph_weight,
                                  model, chunks, emb, diag=diag)
            ok = set(it.get("acceptable_paths", []))
            negatives.append({
                "id": it["id"], "query": it["query"], "top_cosine": diag.get("top_cosine"),
                # retrieval always returns k results, so "returned nothing" cannot
                # happen here; a negative is judged by whether what ranks on top is
                # acceptable. True abstention is a downstream call built on the
                # cosine separation reported below.
                "top1_acceptable": bool(ranked) and ranked[0] in ok,
                "any5_acceptable": any(p in ok for p in ranked[:5]),
                "top5": ranked[:5],
            })
            continue
        gold = [p for p in it["gold_paths"] if p in index_paths]
        if not gold:
            stale.append(it["id"])
            continue
        diag = {}
        ranked = ranked_paths(it["query"], a.mode, a.rerank, a.graph, a.graph_weight, model, chunks, emb, diag=diag)
        row = {"id": it["id"], "source": it.get("source", ""), "category": cat,
               "query": it["query"], "n_gold": len(gold),
               "top_cosine": diag.get("top_cosine"),
               # mega-note contamination: giants crowding the top-5 without being gold
               "mega_in_top5": sum(1 for p in ranked[:5] if p in mega and p not in gold)}
        for k in KS:
            hit = len(set(ranked[:k]) & set(gold))
            r = hit / len(gold)
            p = hit / k
            row[f"recall@{k}"] = round(r, 4)
            row[f"f1@{k}"] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
        rr = 0.0
        for rank, path in enumerate(ranked, 1):
            if path in gold:
                rr = 1.0 / rank
                break
        row["mrr@10"] = round(rr, 4)
        per_item.append(row)

    agg = {"mode": a.mode, "rerank": a.rerank, "graph": a.graph, "graph_weight": a.graph_weight,
           "items_scored": len(per_item), "stale": stale,
           # index provenance — without this a --compare silently attributes CORPUS
           # DRIFT to the change under test (learned live: a metadata writeback
           # re-embedded the corpus between two runs and moved every number)
           "index": {"model": man.get("model"), "chunks": man.get("count"),
                     "notes": man.get("notes"), "built_iso": man.get("built_iso"),
                     "chunker": man.get("chunker")},
           "benchmark_set": os.path.basename(bset)}
    if not per_item:
        sys.exit("FATAL: no positive items scored (all stale or all negatives) — "
                 "nothing to aggregate.")
    for k in KS:
        agg[f"recall@{k}"] = round(sum(r[f"recall@{k}"] for r in per_item) / len(per_item), 4)
        agg[f"f1@{k}"] = round(sum(r[f"f1@{k}"] for r in per_item) / len(per_item), 4)
    agg["mrr@10"] = round(sum(r["mrr@10"] for r in per_item) / len(per_item), 4)

    if is_v2:
        # per-class breakout — the whole point of v2: a macro-average hides a
        # class sitting at zero while the rest are fine.
        by_class = {}
        for cat in sorted({r["category"] for r in per_item}):
            rows = [r for r in per_item if r["category"] == cat]
            by_class[cat] = {"n": len(rows),
                             "recall@1": round(sum(r["recall@1"] for r in rows) / len(rows), 4),
                             "recall@5": round(sum(r["recall@5"] for r in rows) / len(rows), 4),
                             "recall@10": round(sum(r["recall@10"] for r in rows) / len(rows), 4),
                             "mrr@10": round(sum(r["mrr@10"] for r in rows) / len(rows), 4),
                             "misses": sum(1 for r in rows if r["recall@10"] == 0)}
        agg["by_class"] = by_class
        agg["mega_note_contamination"] = round(
            sum(r["mega_in_top5"] for r in per_item) / (5.0 * len(per_item)), 4)
        if negatives:
            pos_cos = sorted(r["top_cosine"] for r in per_item if r["top_cosine"] is not None)
            neg_cos = sorted(n["top_cosine"] for n in negatives if n["top_cosine"] is not None)
            # AUC = P(a random positive scores above a random negative); .5 = no signal
            # sparse mode has no cosine signal: report AUC as unavailable, not 0.0
            auc = None
            if pos_cos and neg_cos:
                auc = (sum(1 for p in pos_cos for n in neg_cos if p > n)
                       + 0.5 * sum(1 for p in pos_cos for n in neg_cos if p == n)) / (len(pos_cos) * len(neg_cos))
            agg["negatives"] = {
                "n": len(negatives),
                "top1_acceptable_rate": round(sum(n["top1_acceptable"] for n in negatives) / len(negatives), 4),
                "any5_acceptable_rate": round(sum(n["any5_acceptable"] for n in negatives) / len(negatives), 4),
                "neg_cosine_median": round(neg_cos[len(neg_cos) // 2], 4) if neg_cos else None,
                "pos_cosine_median": round(pos_cos[len(pos_cos) // 2], 4) if pos_cos else None,
                "neg_cosine_p90": round(neg_cos[int(len(neg_cos) * 0.9)], 4) if neg_cos else None,
                "pos_cosine_p10": round(pos_cos[int(len(pos_cos) * 0.1)], 4) if pos_cos else None,
                "separation_auc": round(auc, 4) if auc is not None else None,
            }

    print(json.dumps(agg, indent=1))
    misses = [r for r in per_item if r["recall@10"] == 0]
    if misses:
        print(f"\nMISSES (recall@10 = 0) — {len(misses)}:")
        for r in misses:
            print(f"  {r['id']} [{r['source']}] {r['query'][:70]}")
    if stale:
        print(f"\nSTALE gold (excluded): {stale}")

    if a.compare:
        old = json.load(open(a.compare))
        oagg = old["aggregate"] if "aggregate" in old else old
        oidx = oagg.get("index")
        if oidx and oidx.get("built_iso") != agg["index"]["built_iso"]:
            print(f"\n⚠️  INDEX DRIFT — comparison is NOT clean: baseline ran on "
                  f"{oidx.get('chunks')} chunks ({oidx.get('built_iso')}, {oidx.get('model')}), "
                  f"this run on {agg['index']['chunks']} ({agg['index']['built_iso']}, "
                  f"{agg['index']['model']}). Part of any delta below is corpus/index change, "
                  f"not the thing you are testing. Re-baseline before drawing conclusions.")
        elif not oidx:
            print("\n⚠️  baseline file predates index-stamping — cannot verify it ran on this "
                  "same index; treat the delta as suggestive only.")
        if oagg.get("benchmark_set") and oagg["benchmark_set"] != agg["benchmark_set"]:
            print(f"⚠️  DIFFERENT BENCHMARK SETS ({oagg['benchmark_set']} vs "
                  f"{agg['benchmark_set']}) — these numbers are not comparable at all.")
        print("\nDELTA vs", a.compare)
        for k in KS:
            key = f"recall@{k}"
            print(f"  {key}: {oagg.get(key)} -> {agg[key]}  ({agg[key] - (oagg.get(key) or 0):+.4f})")
        print(f"  mrr@10: {oagg.get('mrr@10')} -> {agg['mrr@10']}  ({agg['mrr@10'] - (oagg.get('mrr@10') or 0):+.4f})")

    if is_v2:
        print("\nPER-CLASS (the v2 payload — a macro-average hides a class at zero):")
        for cat, v in agg["by_class"].items():
            print(f"  {cat:15s} n={v['n']:3d}  r@1 {v['recall@1']:.3f}  r@5 {v['recall@5']:.3f}  "
                  f"r@10 {v['recall@10']:.3f}  MRR {v['mrr@10']:.3f}  misses {v['misses']}")
        if negatives:
            ng = agg["negatives"]
            print(f"\nNEGATIVES n={ng['n']}: top1-acceptable {ng['top1_acceptable_rate']:.2f} · "
                  f"any5-acceptable {ng['any5_acceptable_rate']:.2f}")
            print(f"  cosine separation AUC {ng['separation_auc'] if ng['separation_auc'] is not None else 'n/a'} "
                  f"(pos median {ng['pos_cosine_median']} vs neg median {ng['neg_cosine_median']}; "
                  f"pos p10 {ng['pos_cosine_p10']} vs neg p90 {ng['neg_cosine_p90']})")
        print(f"\nmega-note contamination in top-5: {agg['mega_note_contamination']:.3f} "
              f"(share of non-gold top-5 slots taken by the {MEGA_N} biggest notes)")

    if a.out:
        json.dump({"aggregate": agg, "per_item": per_item, "negatives": negatives},
                  open(a.out, "w"), indent=1)
        print(f"\nresults -> {a.out}")


if __name__ == "__main__":
    main()
