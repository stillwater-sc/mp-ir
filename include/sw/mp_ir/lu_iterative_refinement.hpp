#pragma once
// mp-ir -- LU-based mixed-precision iterative refinement.
//
// mp-ir is thin: the reusable, Universal-free LU-IR core lives in MTL5
// (mtl::lu_iterative_refine and mtl::normwise_backward_error, in
// mtl/operation/lu_iterative_refine.hpp and mtl/operation/backward_error.hpp).
// This header re-exports that core under the sw::mp_ir namespace so the
// experiments here drive MTL5's dense LU with Universal number systems as the
// working (factor) and residual precisions -- the MTL5 + Universal coupling
// belongs in this composition layer, not in MTL5.
//
// Model (MTL5 core): A, b, x are in the (higher) Residual precision; the cheap
// factorization runs in the explicit lower Working precision:
//
//     mtl::lu_iterative_refine<Working>(A, b, x, opt)
//
// factors A once in Working precision, then corrects x with a residual formed
// in Residual precision.

#include <mtl/operation/lu_iterative_refine.hpp>
#include <mtl/operation/backward_error.hpp>

namespace sw::mp_ir {

// Re-export the MTL5 core under sw::mp_ir.
using mtl::lu_iterative_refine;
using mtl::lu_refine_options;
using mtl::lu_refine_result;
using mtl::normwise_backward_error;

// Friendly aliases for the composition layer.
using ir_options = mtl::lu_refine_options;
using ir_result  = mtl::lu_refine_result;

} // namespace sw::mp_ir
