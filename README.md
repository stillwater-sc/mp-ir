# mp-ir

[![CMake](https://github.com/stillwater-sc/mp-ir/actions/workflows/cmake.yml/badge.svg)](https://github.com/stillwater-sc/mp-ir/actions/workflows/cmake.yml)

**Mixed-precision iterative refinement.** mp-ir composes two header-only
libraries — [MTL5](https://github.com/stillwater-sc/mtl5) for linear algebra and
[Universal](https://github.com/stillwater-sc/universal) for parameterized number
systems — to study iterative refinement: factor a system in a low working
precision, then recover accuracy by correcting the solution with a
higher-precision residual (half precision, posits, ... as the working type).

MTL5 deliberately has **no dependency on Universal**: it is the general
linear-algebra layer, and it already ships a Universal-free sparse
`iterative_refine` core. mp-ir is the integration layer where MTL5's dense LU
and Universal's number types meet — the mixed-precision *experiments* live here.

## Build

```bash
# Dependencies (MTL5 + Universal) are pulled automatically via FetchContent.
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the smoke test
ctest --test-dir build --output-on-failure

# Run the precision study (optional arg: problem size)
./build/applications/dense_lu/lu_ir_precision/lu_ir_precision 8

# Run the benchmark (optional arg: problem size)
./build/benchmarks/dense_lu/lu_ir_benchmark 64
```

Using local checkouts instead of fetching from GitHub:

```bash
cmake -B build \
  -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 \
  -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=../universal
```

## Layout

The repo is organized by iterative-refinement category: **dense_lu** (LU factor
+ refine), **scaling** (round-and-replace / scale-and-round squeeze strategies),
and **sparse** (sparse direct factor + refine, driving MTL5's core).

```
include/sw/mp_ir/
  lu_iterative_refinement.hpp    # re-exports MTL5 core (mtl::lu_iterative_refine)
include/mtl/math/
  quire_accumulator.hpp          # MTL5 accumulator_traits <- Universal quire bridge
applications/
  dense_lu/lu_ir_precision/      # refined-vs-unrefined residual across precisions
benchmarks/
  dense_lu/                      # factor/refine timing + steps-to-floor
tests/
  dense_lu/                      # LU-IR smoke test + migrated luir experiment
  scaling/                       # squeeze preconditioners (round/scale/two-sided)
  sparse/                        # (no tests yet -- see docs/roadmap.md)
docs/roadmap.md                  # milestones and migration plan
```

## License

MIT — see [LICENSE](LICENSE).
