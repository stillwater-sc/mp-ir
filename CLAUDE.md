# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

mp-ir is the **integration layer** for mixed-precision iterative refinement. It
composes two header-only sister libraries:

- [MTL5](https://github.com/stillwater-sc/mtl5) — C++20 linear algebra.
- [Universal](https://github.com/stillwater-sc/universal) — parameterized number
  systems (`cfloat`, `posit`, ...).

**Architectural rule:** MTL5 is the general linear-algebra layer and MUST NOT
depend on Universal. MTL5 already provides a Universal-free sparse
`iterative_refine` core; all MTL5 + Universal coupling lives here in mp-ir.

## Build Commands

```bash
# Dependencies are pulled automatically via FetchContent.
cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build build -j
ctest --test-dir build --output-on-failure

# Use local sister checkouts instead of fetching from GitHub:
cmake -B build -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 \
               -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=../universal
```

## Architecture

- Header-only composition under `include/sw/mp_ir/`. Namespace: `sw::mp_ir`.
  `lu_iterative_refinement.hpp` drives MTL5's dense `lu_factor`/`lu_solve` with a
  low working precision and a higher-precision residual.
- CMake: INTERFACE library `sw::mp_ir` linking MTL5 + Universal. Options:
  `MPIR_BUILD_APPLICATIONS`, `MPIR_BUILD_TESTS`, `MPIR_BUILD_BENCHMARKS`.
- `applications/`, `benchmarks/`, and `tests/` are organized by
  iterative-refinement category: `dense_lu/` (LU factor + refine), `scaling/`
  (round-and-replace / scale-and-round squeeze strategies), `sparse/` (sparse
  direct factor + refine).
- `applications/` — demonstration programs (each its own CMakeLists).
- `benchmarks/` — accuracy/performance studies (factor vs refine cost, steps to
  the residual floor).
- `tests/` — lightweight self-checking executables (no external framework);
  register with `mpir_add_test`.
- `docs/roadmap.md` — milestones and the Universal->mp-ir migration plan.

## Conventions

- C++20, header-only. Match the sister repos (mtl5, mp-iterative) for style and
  CMake structure.
- Conventional Commits. Feature branches + PRs to `main`; CI must pass.
- Never commit build artifacts or downloaded matrices.
