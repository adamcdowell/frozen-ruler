"""Exact tests for the hand-rolled statistics in arm_compare.py.

  uv run python3 test_stats.py

No dependencies. Values checked against closed-form binomial arithmetic and
published Wilson-interval reference values.
"""
import math
import arm_compare as ac


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def rows(vals, metric="recall@1"):
    return {i: {metric: v} for i, v in enumerate(vals)}


# --- sign test: exact two-sided binomial ---
# 9 up, 0 down -> p = 2 * (1/2)^9 = 0.00390625 (the graph-kill shape)
up, down, p = ac.sign_test(rows([0] * 9), rows([1] * 9), "recall@1")
assert (up, down) == (9, 0)
approx(p, 2 * (0.5 ** 9))

# 1 up, 1 down -> p = 1.0 (capped)
up, down, p = ac.sign_test(rows([0, 1]), rows([1, 0]), "recall@1")
assert (up, down) == (1, 1)
approx(p, 1.0)

# no discordant pairs -> p = 1.0 by convention
up, down, p = ac.sign_test(rows([1, 1]), rows([1, 1]), "recall@1")
assert (up, down) == (0, 0)
approx(p, 1.0)

# 7 up, 3 down -> two-sided binomial tail: 2 * sum C(10,k)/2^10 for k=7..10
expect = 2 * sum(math.comb(10, k) for k in range(7, 11)) / 2 ** 10
up, down, p = ac.sign_test(rows([0] * 7 + [1] * 3), rows([1] * 7 + [0] * 3), "recall@1")
approx(p, expect)

# sub-epsilon differences are ties, not wins
up, down, p = ac.sign_test(rows([0.5]), rows([0.5 + 1e-12]), "recall@1")
assert (up, down) == (0, 0)

# --- Wilson 95% interval ---
# n=0 -> (0, 0) guard
assert ac.wilson(0, 0) == (0.0, 0.0)

# reference value: 8/10 successes, z=1.96 -> (0.4902, 0.9433) (4 dp,
# hand-computed: center 0.71674, half-width 0.22658)
lo, hi = ac.wilson(8, 10)
approx(round(lo, 4), 0.4902, 1e-9)
approx(round(hi, 4), 0.9433, 1e-9)

# 0/10 lower bound clamps to 0; 10/10 upper clamps to 1
lo, hi = ac.wilson(0, 10)
assert lo == 0.0 and 0 < hi < 0.4
lo, hi = ac.wilson(10, 10)
assert hi == 1.0 and 0.6 < lo < 1

# interval always contains the point estimate
for h, n in [(1, 3), (5, 9), (189, 209)]:
    lo, hi = ac.wilson(h, n)
    assert lo <= h / n <= hi

print("test_stats: all assertions passed")
