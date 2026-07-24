#pragma once
// mp-ir -- shared driver for the squeeze/scaling LU-IR studies.
//
// For a given squeeze strategy, sweep a set of low (factorization) precisions
// and, for each, compare a plain low-precision LU-IR solve against a squeezed
// one (the matrix conditioned into the low precision's range first). Reports
// iterations and forward error ||X - x||_inf against the known all-ones
// solution. Migrated from Universal applications/performance/ir (round-and-
// replace / scale-and-round / two-sided), re-expressed on MTL5's LU-IR core.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/generators/minij.hpp>
#include <mtl/generators/lehmer.hpp>
#include <mtl/generators/hilbert.hpp>

#include <sw/mp_ir/lu_iterative_refinement.hpp>
#include <sw/mp_ir/squeeze.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace sw::mp_ir::study {

// double is the storage + residual precision: it converts to/from both posit and
// cfloat low precisions (Universal has no direct posit<->cfloat conversion).
using High = double;

enum class squeeze_kind { none, round_and_replace, scale_and_round, two_sided };

// materialize a named reference matrix (double), scaled by `scale`
inline mtl::mat::dense2D<double> reference(const std::string& name, std::size_t n, double scale) {
    mtl::mat::dense2D<double> A(n, n);
    auto fill = [&](auto gen) { for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) A(i, j) = scale * gen(i, j); };
    if      (name == "minij")   fill(mtl::generators::minij<double>(n));
    else if (name == "lehmer")  fill(mtl::generators::lehmer<double>(n));
    else if (name == "hilbert") fill(mtl::generators::hilbert<double>(n));
    else if (name == "graded") {
        // a diagonally dominant system with wildly varying ROW magnitudes (row i
        // scaled by 100^(i/(n-1))), so one-sided scaling cannot fix it but
        // two-sided (row+column) equilibration can.
        mtl::generators::minij<double> g(n);
        for (std::size_t i = 0; i < n; ++i) {
            double rs = std::pow(100.0, double(i) / double(n > 1 ? n - 1 : 1));
            for (std::size_t j = 0; j < n; ++j) A(i, j) = scale * rs * (g(i, j) + (i == j ? double(n) : 0.0));
        }
    }
    else                        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j)
                                    A(i, j) = scale * ((i == j) ? double(n) : 1.0 / double(i + j + 2));
    return A;
}

// run LU-IR (factor in Low) on a High matrix; returns {iters, forward error}.
template <typename Low>
inline std::pair<int, double> solve(const mtl::mat::dense2D<High>& A) {
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<High> b(n, High(0)), x;
    for (std::size_t i = 0; i < n; ++i) { High s(0); for (std::size_t j = 0; j < n; ++j) s = s + A(i, j); b[i] = s; }
    ir_options opt; opt.max_iter = 20;
    auto res = lu_iterative_refine<Low>(A, b, x, opt);
    double fe = 0.0;
    for (std::size_t i = 0; i < n; ++i) { double d = double(x[i]) - 1.0; if (std::isfinite(d)) fe = std::max(fe, std::abs(d)); else fe = 1e300; }
    return { res.iters, fe };
}

// apply the selected squeeze to a High matrix in place
template <typename Low>
inline void squeeze(mtl::mat::dense2D<High>& A, squeeze_kind kind, High T) {
    switch (kind) {
    case squeeze_kind::round_and_replace: round_and_replace<Low>(A); break;
    case squeeze_kind::scale_and_round:   scale_and_round<Low>(A, T); break;
    case squeeze_kind::two_sided: {
        mtl::vec::dense_vector<High> R, S;
        two_sided_scale_and_round<Low>(A, T, R, S);
        break;
    }
    case squeeze_kind::none: default: break;
    }
}

// one row of the study for a given Low precision
template <typename Low>
inline void row(const std::string& label, const mtl::mat::dense2D<double>& ref, squeeze_kind kind, High T) {
    const std::size_t n = ref.num_rows();
    mtl::mat::dense2D<High> A(n, n), Asq(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) { A(i, j) = High(ref(i, j)); Asq(i, j) = High(ref(i, j)); }
    squeeze<Low>(Asq, kind, T);

    auto plain    = solve<Low>(A);
    auto squeezed = solve<Low>(Asq);

    auto cell = [](std::pair<int, double> r) {
        std::ostringstream os;
        os << "it=" << std::setw(2) << r.first << " fe=" << std::scientific << std::setprecision(1) << r.second;
        return os.str();
    };
    std::cout << "  " << std::left << std::setw(14) << label
              << "  plain[" << cell(plain) << "]   squeezed[" << cell(squeezed) << "]\n";
}

// the full study: sweep low precisions on a (scaled) named matrix
inline void run(const std::string& squeeze_name, squeeze_kind kind,
                const std::string& matrix, std::size_t n, double scale, High T) {
    using namespace sw::universal;
    auto ref = reference(matrix, n, scale);
    std::cout << squeeze_name << " squeeze -- matrix " << matrix << " x" << scale
              << " (" << n << "x" << n << "), High = double, T = " << double(T) << "\n";
    std::cout << "  low precision   plain LU-IR                squeezed LU-IR\n";
    row<posit<8, 2>>  ("posit<8,2>",  ref, kind, T);
    row<posit<12, 2>> ("posit<12,2>", ref, kind, T);
    row<posit<16, 2>> ("posit<16,2>", ref, kind, T);
    row<posit<16, 5>> ("posit<16,5>", ref, kind, T);
    row<cfloat<16, 5, std::uint16_t, true, false, false>>("cfloat<16,5>", ref, kind, T);
    row<cfloat<8, 4, std::uint8_t, true, false, false>>  ("cfloat<8,4>",  ref, kind, T);
}

} // namespace sw::mp_ir::study
