#pragma once
// mp-ir -- LU-based mixed-precision iterative refinement.
//
// Iterative refinement solves A x = b by factoring A once in a (low) working
// precision, then repeatedly correcting the solution with a residual formed in
// a higher precision:
//
//     factor  A = P L U            (working precision)
//     solve   x0 = U\(L\(P b))     (working precision)
//     repeat  r  = b - A x         (residual precision, then cast to working)
//             d  = U\(L\(P r))     (working precision)
//             x  = x + d
//
// The working-precision factorization is cheap; the higher-precision residual
// is what lets the refined solution reach an accuracy far better than the
// working precision's own solve. This is the canonical mixed-precision kernel
// (Wilkinson; Higham, "Accuracy and Stability of Numerical Algorithms").
//
// This header is the composition-layer home for the *experiments*: it drives
// MTL5's Universal-free dense LU (mtl::lu_factor / mtl::lu_solve) with Universal
// number systems as the working / residual precisions. Per the ecosystem rule,
// MTL5 never depends on Universal; that coupling lives here in mp-ir.

#include <cmath>
#include <cstddef>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>

namespace sw::mp_ir {

/// Outcome of an iterative-refinement solve.
struct ir_result {
    std::size_t iterations   = 0;      ///< refinement steps actually taken
    bool        converged    = false;  ///< reached the tolerance (if one was set)
    double      final_residual = 0.0;  ///< best ||b - A x||_inf, measured in double
};

/// Refinement controls.
struct ir_options {
    std::size_t max_iterations = 20;   ///< cap on refinement steps
    double      tolerance      = 0.0;  ///< stop when the double residual <= tolerance;
                                       ///< 0 runs the full budget and returns the best iterate
};

/// Solve A x = b by LU iterative refinement.
///   WorkingType  -- precision of the LU factorization and correction solves
///                   (deduced from the arguments).
///   ResidualType -- (usually higher) precision in which r = b - A x is formed;
///                   defaults to double.
/// A and b are supplied in WorkingType; the refined x is returned in WorkingType.
template <typename ResidualType = double, typename WorkingType>
ir_result lu_ir_solve(const mtl::mat::dense2D<WorkingType>& A,
                      const mtl::vec::dense_vector<WorkingType>& b,
                      mtl::vec::dense_vector<WorkingType>& x,
                      const ir_options& opts = {}) {
    using Mat       = mtl::mat::dense2D<WorkingType>;
    using Vec       = mtl::vec::dense_vector<WorkingType>;
    using size_type = typename Mat::size_type;
    const std::size_t n = A.num_rows();

    // ||b - A x||_inf, accumulated in double for a precision-independent yardstick
    auto residual_inf = [&](const Vec& xx) {
        double rmax = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double ax = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                ax += static_cast<double>(A(i, j)) * static_cast<double>(xx[j]);
            rmax = std::max(rmax, std::abs(static_cast<double>(b[i]) - ax));
        }
        return rmax;
    };

    // factor a copy of A in working precision, then take the initial solve
    Mat LU(A);
    std::vector<size_type> pivot;
    mtl::lu_factor(LU, pivot);
    x = Vec(n, WorkingType(0));
    mtl::lu_solve(LU, pivot, x, b);

    ir_result res;
    double best = residual_inf(x);
    Vec    bestx = x;

    for (std::size_t it = 0; it < opts.max_iterations; ++it) {
        // r = b - A x, accumulated in ResidualType, then cast down to working
        Vec r(n, WorkingType(0));
        for (std::size_t i = 0; i < n; ++i) {
            ResidualType ax = ResidualType(0);
            for (std::size_t j = 0; j < n; ++j)
                ax += ResidualType(A(i, j)) * ResidualType(x[j]);
            r[i] = WorkingType(ResidualType(b[i]) - ax);
        }
        // correction and update
        Vec d(n, WorkingType(0));
        mtl::lu_solve(LU, pivot, d, r);
        for (std::size_t i = 0; i < n; ++i) x[i] = x[i] + d[i];

        ++res.iterations;
        double cur = residual_inf(x);
        if (cur < best) { best = cur; bestx = x; }
        if (opts.tolerance > 0.0 && cur <= opts.tolerance) break;
    }

    x = bestx;                                   // return the best iterate seen
    res.final_residual = best;
    res.converged = (opts.tolerance > 0.0) ? (best <= opts.tolerance) : true;
    return res;
}

} // namespace sw::mp_ir
