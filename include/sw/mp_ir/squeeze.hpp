#pragma once
// mp-ir -- matrix "squeeze" preconditioners for low-precision LU factorization.
//
// Migrated from Universal blas/squeeze.hpp (author James Quinlan). Before an
// LU-IR run factors A in a low precision, these map the (high-precision) matrix
// into the low precision's representable range so the factorization does not
// overflow or lose the whole matrix to a bad dynamic range:
//
//   round_and_replace      -- clamp |A| to the low precision's maxpos, so the
//                             cast to Low produces no overflow/infinities;
//   scale_and_round        -- scale A by mu = T / max|A| so its largest element
//                             lands near the low precision's range (posit
//                             convention), then the caller rounds to Low;
//   two_sided_scale_and_round -- row + column equilibration (R = 1/max_row,
//                             S = 1/max_col) followed by scale_and_round.
//
// Each operates in place on the high-precision A. Because an LU-IR study derives
// the right-hand side from the (already squeezed) A, the exact solution is
// unchanged -- the squeeze only conditions the low-precision factorization.
//
// These live in mp-ir, not MTL5: choosing the scale from the LOW precision's
// maxpos couples to Universal number systems, which MTL5 must not depend on.

#include <cstddef>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <universal/number/shared/specific_value_encoding.hpp>   // SpecificValue

namespace sw::mp_ir {

namespace detail {
    template <typename T> T mag(const T& x) { return x < T(0) ? T(-x) : x; }
}

/// Clamp |A(i,j)| to the low precision's maxpos so casting A to `Low` cannot
/// overflow to infinity (round-and-replace). High-precision, in place.
template <typename Low, typename High, typename P>
void round_and_replace(mtl::mat::dense2D<High, P>& A) {
    const High maxpos = High(Low(sw::universal::SpecificValue::maxpos));
    const std::size_t m = A.num_rows(), n = A.num_cols();
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            High v = A(i, j);
            if (detail::mag(v) > maxpos) A(i, j) = (v < High(0)) ? -maxpos : maxpos;
        }
}

/// Scale A by mu = T / max|A| (posit convention) so the largest element maps
/// near the low precision's range; the caller rounds to `Low`. Returns mu.
/// (`Low` is only part of the API for parity with the two-sided variant and the
/// original cfloat convention mu = T*maxpos/max|A|.)
template <typename Low, typename High, typename P>
High scale_and_round(mtl::mat::dense2D<High, P>& A, High T) {
    (void)sizeof(Low);
    const std::size_t m = A.num_rows(), n = A.num_cols();
    High Amax(0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) { High a = detail::mag(A(i, j)); if (a > Amax) Amax = a; }
    High mu = (Amax > High(0)) ? T / Amax : High(1);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = mu * A(i, j);
    return mu;
}

/// Row + column equilibration (R = 1/max_row, S = 1/max_col) followed by
/// scale_and_round. Returns mu; fills the row/column scalers R and S.
template <typename Low, typename High, typename P>
High two_sided_scale_and_round(mtl::mat::dense2D<High, P>& A, High T,
                               mtl::vec::dense_vector<High>& R,
                               mtl::vec::dense_vector<High>& S) {
    const std::size_t n = A.num_rows();
    R = mtl::vec::dense_vector<High>(n, High(1));
    S = mtl::vec::dense_vector<High>(n, High(1));
    // row equilibration
    for (std::size_t i = 0; i < n; ++i) {
        High M(0);
        for (std::size_t j = 0; j < n; ++j) { High a = detail::mag(A(i, j)); if (a > M) M = a; }
        R[i] = (M > High(0)) ? High(1) / M : High(1);
        for (std::size_t j = 0; j < n; ++j) A(i, j) = R[i] * A(i, j);
    }
    // column equilibration
    for (std::size_t j = 0; j < n; ++j) {
        High M(0);
        for (std::size_t i = 0; i < n; ++i) { High a = detail::mag(A(i, j)); if (a > M) M = a; }
        S[j] = (M > High(0)) ? High(1) / M : High(1);
        for (std::size_t i = 0; i < n; ++i) A(i, j) = S[j] * A(i, j);
    }
    return scale_and_round<Low>(A, T);
}

} // namespace sw::mp_ir
