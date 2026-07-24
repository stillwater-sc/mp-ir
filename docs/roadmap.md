# mp-ir roadmap

## Milestone 0: composition layer bootstrapped (done)

- CMake scaffold replicated from [mp-iterative](https://github.com/stillwater-sc/mp-iterative):
  INTERFACE library `sw::mp_ir`, find_package → FetchContent fallback for MTL5 +
  Universal, config-package install, CI matrix (MSVC/GCC/Clang/AppleClang).
- Thin driver `include/sw/mp_ir/lu_iterative_refinement.hpp`: re-exports MTL5's
  Universal-free core `mtl::lu_iterative_refine<Working>` (factor in a low
  Working precision, refine with a higher-precision residual) and
  `mtl::normwise_backward_error` under `sw::mp_ir`. The reusable algorithm lives
  in MTL5; mp-ir supplies the Universal number systems and experiments.
- Smoke test: LU-IR on a diagonally dominant dense system stored in `double`
  with the factorization in `double` / `float` / `cfloat<16,5>` / `posit<16,2>`
  — refined residual beats the plain low-precision solve.
- Demo application `lu_ir_precision` (refined-vs-unrefined table) and benchmark
  `lu_ir_benchmark` (factor/refine timing + steps to floor).
- Repo organized by refinement category: `dense_lu/` (LU factor + refine),
  `scaling/` (round-and-replace / scale-and-round), `sparse/` (sparse direct +
  refine).
- `include/mtl/math/quire_accumulator.hpp`: the MTL5 `accumulator_traits` ←
  Universal quire bridge (the coupling that must not live in MTL5 itself), for
  forming the residual in an exact quire.

## Milestone 1: migrate the Universal iterative-refinement study

Migrate from Universal `applications/performance/ir/` (author James Quinlan),
re-expressed on the MTL5 dense LU core + Universal precisions:

- `luir` — the three-precision (High/Working/Low) LU iterative-refinement
  driver (Universal `ext/solvers/luir.hpp` `SolveIRLU<High,Working,Low>`); the
  Universal-free driver core lands in MTL5, the posit/precision experiments here.
- `roundAndReplace`, `scaleAndRound`, `twoSidedScaleAndRound` — the squeeze /
  scaling strategies (Universal `blas/squeeze.hpp`) under `scaling/`.
- Normwise backward error (`nbe`, Higham Thm 7.1) as the convergence/quality
  metric.
- Reproduce the current Universal `luir.cpp` results as the acceptance check.

## Milestone 2: mixed-precision refinement studies

- Precision of the residual vs the factorization: how low can the working
  precision go before refinement stalls, per condition number.
- Quire (exact dot product) accumulation of the residual r = b - A x via the
  `accumulator_traits` bridge, vs naive / fma residuals.
- Sparse direct factor + refinement: drive MTL5's Universal-free
  `sparse/iterative_refine.hpp` with Universal working/residual precisions
  (`sparse/` category), connecting with mp-spice's KLU work.
- SuiteSparse / Matrix Market driver for realistically conditioned problems.
