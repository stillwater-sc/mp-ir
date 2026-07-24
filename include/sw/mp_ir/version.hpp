#pragma once
// mp-ir -- mixed-precision iterative refinement (MTL5 + Universal)
//
// Header-only composition layer. Shared iterative-refinement utilities live
// under sw::mp_ir (see lu_iterative_refinement.hpp). This header carries the
// version metadata.

namespace sw::mp_ir {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 1;
inline constexpr int version_patch = 0;

} // namespace sw::mp_ir
