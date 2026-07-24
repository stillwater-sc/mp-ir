# Dense LU iterative refinement.

# Smoke test: LU-IR solves through the MTL5 + Universal composition and the
# refined solution reaches an accuracy better than the working precision's own
# solve, across number types.
mpir_add_test(test_lu_ir_smoke)
