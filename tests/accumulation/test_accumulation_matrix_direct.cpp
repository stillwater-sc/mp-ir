// mp-ir test: the configuration-3 (quire) direct-solver and refinement rows of
// the accumulation validation matrix, stillwater-sc/mtl5#586 (issue #11). Each
// row names the MTL5 tag it mirrors, all in mtl5
// tests/unit/operation/test_accumulation_solvers_mt.cpp:
//
//   [586:direct]  dense and sparse LU, Cholesky, LDLT, inv, Bunch-Kaufman, QR
//   [586:refine]  supernodal_lu_solve_refined, float factor, double residual
//
// Configurations, at element type float:
//   1 plain (Accumulator = float), 2 mtl::math::fma_accumulator<float>,
//   3 sw::universal::quire<float>.
// MTL5 measured these rows with a test-only Kulisch accumulator standing in for
// configuration 3. A quire is correctly rounded too, so its residuals should be
// MTL5's to the digit; the inputs (Hilbert(n) rounded to float, b = A x for a
// fixed x) are the same. The relative residual ||b - A x|| / ||b|| is computed
// exactly, with mtl::math::exact_accumulator. Every row also runs that Kulisch
// accumulator as a configuration and asserts the quire's residual equals it.
//
// Measured (the test prints every row; GCC and Clang agree), Hilbert(n) in float:
//
//                                 plain     fma       quire
//   sparse LU         n = 5       2.4e-8    1.5e-8    1.1e-9
//                     n = 8       1.5e-8    3.5e-8    2.9e-9
//   sparse Cholesky   n = 5       3.0e-8    2.1e-8    7.2e-9
//                     n = 8       2.6e-8    1.3e-8    1.1e-9
//   dense LU          n = 5       1.8e-8    8.5e-9    1.1e-9
//                     n = 8       2.8e-8    1.3e-8    2.9e-9
//   dense Cholesky    as sparse Cholesky, to the digit
//   dense inv         n = 5       1.4e-3    3.1e-4    2.5e-5
//                     n = 8       1.7e-1    6.9e-1    3.8e-3
//   LDLT, dense and   n = 6       2.0e-8    1.3e-8    1.2e-8   (the closest: 1.06x)
//   sparse            n = 8       2.9e-8    1.4e-8    3.7e-9
//   sparse QR         n = 8       1.9e-7    1.1e-7    5.0e-7   (no ordering)
//   Bunch-Kaufman     Hilbert(6)  4.9e-8    1.1e-8    1.5e-8   (no ordering)
//                     saddle n=7  9.4e-8    2.3e-6    8.0e-8
//   Cholesky, Hilbert(9)          not positive definite  1.2e-8
//   refinement        n = 9       1.2e-4    3.1e-4    1.1e-12
//   (float factor,    n = 10      1.6e-4    5.1e-4    7.3e-12
//   double residual)  n = 11      5.3e-4    3.0e-4    6.8e-13
//
// Every figure MTL5 records for these rows is reproduced, so the quire and
// MTL5's Kulisch stand-in agree. One correction to MTL5's prose: refinement
// with configuration 3 reaches 7.3e-12 at n = 10, not "1.1e-12 or better"
// (its assertion, < 1e-10, holds).
//
// Built with -ffp-contract=off (tests/accumulation/CMakeLists.txt). Returns
// non-zero on failure.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <universal/number/quire/quire.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/exact_accumulator.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/ldlt_bk.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace {

using quire_f = sw::universal::quire<float>;
using exact_f = mtl::math::exact_accumulator<float>;
using fma_f   = mtl::math::fma_accumulator<float>;
using smat    = mtl::mat::compressed2D<float>;
using fvec    = mtl::vec::dense_vector<float>;
namespace sf  = mtl::sparse::factorization;

// The quire takes the error-free split of LDLT's three-factor terms (mtl5#602).
static_assert(mtl::math::splits_product_terms_v<quire_f, float>);

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

/// The Hilbert matrix, rounded to float, stored as T.
template <typename T>
mtl::mat::compressed2D<T> hilbert(std::size_t n) {
    mtl::mat::compressed2D<T> A(n, n);
    {
        mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                ins[i][j] << static_cast<T>(static_cast<float>(1.0 / static_cast<double>(i + j + 1)));
    }
    return A;
}

/// The saddle-point matrix [[H, 1], [1^T, 0]], H = Hilbert(n): symmetric
/// indefinite, so Bunch-Kaufman interchanges and takes 2x2 pivots.
smat saddle_hilbert(std::size_t n) {
    smat A(n + 1, n + 1);
    {
        mtl::mat::inserter<smat> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j)
                ins[i][j] << static_cast<float>(1.0 / static_cast<double>(i + j + 1));
            ins[i][n] << 1.0f;
            ins[n][i] << 1.0f;
        }
    }
    return A;
}

/// ||b - A x||_2 / ||b||_2, each residual entry computed exactly.
double rel_residual(const smat& A, const fvec& x, const fvec& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& v = A.ref_data();
    double s = 0.0, sb = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        mtl::math::exact_accumulator<float> a;
        accumulator_clear(a);
        accumulator_add_product(a, b(r), 1.0f);
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            accumulator_add_product(a, -v[k], x(ci[k]));
        const double e = accumulator_value(a, std::type_identity<double>{});
        s += e * e;
        sb += static_cast<double>(b(r)) * static_cast<double>(b(r));
    }
    return std::sqrt(s / sb);
}

/// b = A * x for a fixed, non-trivial x (MTL5's rhs()).
fvec rhs(const smat& A) {
    fvec x(A.num_rows()), b(A.num_rows());
    for (std::size_t i = 0; i < x.size(); ++i)
        x(i) = static_cast<float>(1 + (i * 7919) % 13) / 13.0f;
    mtl::mult<double>(A, x, b);
    return b;
}

template <typename Acc, typename Solve>
double direct_residual(const smat& A, Solve solve) {
    const fvec b = rhs(A);
    fvec x(A.num_rows(), 0.0f);
    solve.template operator()<Acc>(A, x, b);
    return rel_residual(A, x, b);
}

mtl::mat::dense2D<float> dense(const smat& A) {
    mtl::mat::dense2D<float> D(A.num_rows(), A.num_cols());
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) D(i, j) = A(i, j);
    return D;
}

struct sparse_lu {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        sf::sparse_lu_solve<Acc>(A, x, b);
    }
};
struct sparse_cholesky {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        sf::sparse_cholesky_solve<Acc>(A, x, b);
    }
};
struct sparse_ldlt {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        sf::sparse_ldlt_solve<Acc>(A, x, b);
    }
};
struct sparse_qr {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        sf::sparse_qr_solve<Acc>(A, x, b);
    }
};
struct dense_lu {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        auto D = dense(A);
        if (mtl::lu_apply<Acc>(D, x, b) != 0) throw std::runtime_error("lu: singular");
    }
};
struct dense_cholesky {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        auto D = dense(A);
        if (mtl::cholesky_factor<Acc>(D) != 0) throw std::runtime_error("cholesky: not positive definite");
        mtl::cholesky_solve<Acc>(D, x, b);
    }
};
struct dense_ldlt {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        auto D = dense(A);
        if (mtl::ldlt_factor<Acc>(D) != 0) throw std::runtime_error("ldlt: zero pivot");
        mtl::ldlt_solve<Acc>(D, x, b);
    }
};
struct dense_ldlt_bk {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        auto D = dense(A);
        mtl::bk_pivot_info piv;
        if (mtl::ldlt_bk_factor<Acc>(D, piv) != 0) throw std::runtime_error("ldlt_bk: singular block");
        mtl::ldlt_bk_solve<Acc>(D, piv, x, b);
    }
};
struct dense_inv {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        const auto X = mtl::inv<Acc>(dense(A));
        mtl::mult<Acc>(X, b, x);   // x = inv(A) b, in the same configuration
    }
};

struct residuals { double plain, fma, quire, exact; };

template <typename Solve>
residuals measure(const char* name, const smat& A, Solve solve) {
    const residuals r{direct_residual<float>(A, solve), direct_residual<fma_f>(A, solve),
                      direct_residual<quire_f>(A, solve), direct_residual<exact_f>(A, solve)};
    std::printf("  %-16s order %2zu   plain %.1e   fma %.1e   quire %.1e\n",
                name, A.num_rows(), r.plain, r.fma, r.quire);
    check(r.quire == r.exact, "[586:direct] the quire's residual is the Kulisch accumulator's");
    return r;
}

/// Configuration 3 at least `factor` below both 1 and 2.
template <typename Solve>
void ordered(const char* name, const smat& A, Solve solve, double factor, const char* what) {
    const auto r = measure(name, A, solve);
    check(factor * r.quire < r.plain && factor * r.quire < r.fma, what);
}

// ---------------------------------------------------------------------------
// [586:direct]
// ---------------------------------------------------------------------------

void direct_rows() {
    std::printf("[586:direct] relative residual, Hilbert(n) in float\n");
    for (std::size_t n = 5; n <= 8; ++n) {
        const auto A = hilbert<float>(n);
        // MTL5 measured: sparse LU and Cholesky at least 2.8x, dense LU 4.3x,
        // dense Cholesky 2.9x, inv 10x below both 1 and 2.
        ordered("sparse LU", A, sparse_lu{}, 2.0, "[586:direct] sparse LU: quire 2x below plain and fma");
        ordered("sparse Cholesky", A, sparse_cholesky{}, 2.0,
                "[586:direct] sparse Cholesky: quire 2x below plain and fma");
        ordered("dense LU", A, dense_lu{}, 2.0, "[586:direct] dense LU: quire 2x below plain and fma");
        ordered("dense Cholesky", A, dense_cholesky{}, 2.0,
                "[586:direct] dense Cholesky: quire 2x below plain and fma");
        ordered("dense inv", A, dense_inv{}, 4.0, "[586:direct] inv: quire 4x below plain and fma");

        // LDLT: lowest since mtl5#602, by as little as 1.06x, so no margin.
        for (const auto& r : {measure("sparse LDLT", A, sparse_ldlt{}), measure("dense LDLT", A, dense_ldlt{})}) {
            check(r.plain < 1e-6 && r.fma < 1e-6, "[586:direct] LDLT: plain and fma solve");
            check(r.quire < r.plain && r.quire < r.fma, "[586:direct] LDLT: quire lowest");
        }

        // Sparse QR: no ordering in MTL5; each configuration solves.
        const auto q = measure("sparse QR", A, sparse_qr{});
        check(q.plain < 1e-6 && q.fma < 1e-6 && q.quire < 1e-6, "[586:direct] sparse QR: every configuration solves");
    }

    // Bunch-Kaufman: no ordering in MTL5; each configuration solves, on Hilbert
    // and on the indefinite saddle, n = 4 .. 9.
    for (std::size_t n = 4; n <= 9; ++n)
        for (const auto& A : {hilbert<float>(n), saddle_hilbert(n)}) {
            const auto r = measure("Bunch-Kaufman", A, dense_ldlt_bk{});
            check(r.plain < 1e-5 && r.fma < 1e-5 && r.quire < 1e-5,
                  "[586:direct] Bunch-Kaufman: every configuration solves");
        }

    // Hilbert(9) is positive definite, but its smallest eigenvalue (~5e-13) is
    // far below float's resolution: summing the column updates with a rounding
    // accumulator drives the last pivot non-positive; summing them exactly does not.
    const auto H9 = hilbert<float>(9);
    auto only_quire_factors = [&](const char* name, auto solve) {
        auto throws = [&](auto tag) {
            try {
                direct_residual<decltype(tag)>(H9, solve);
            } catch (const std::runtime_error&) {
                return true;
            }
            return false;
        };
        check(throws(float{}), "[586:direct] Hilbert(9) Cholesky: plain fails to factor");
        check(throws(fma_f{}), "[586:direct] Hilbert(9) Cholesky: fma fails to factor");
        double r3 = 1.0;
        try {
            r3 = direct_residual<quire_f>(H9, solve);
        } catch (const std::runtime_error&) {
            check(false, "[586:direct] Hilbert(9) Cholesky: quire factors");
        }
        std::printf("  %-16s Hilbert(9)   plain and fma not positive definite   quire %.1e\n", name, r3);
        check(r3 < 1e-7, "[586:direct] Hilbert(9) Cholesky: quire solves");
    };
    only_quire_factors("sparse Cholesky", sparse_cholesky{});
    only_quire_factors("dense Cholesky", dense_cholesky{});
}

// ---------------------------------------------------------------------------
// [586:refine]
// ---------------------------------------------------------------------------

void refine_rows() {
    // Factor in float with each accumulator, refine the residual in double. The
    // rounding accumulators leave a factor too poor for refinement to contract
    // on Hilbert(9..11); the quire does not. MTL5 measured: 1 and 2 stall between
    // 1.2e-4 and 5.3e-4, 3 reaches 1.1e-12 or better.
    std::printf("[586:refine] float factor, double residual, relative residual after refinement\n");
    for (std::size_t n = 9; n <= 11; ++n) {
        const auto A = hilbert<double>(n);
        auto refine = [&](auto tag) {
            using Acc = decltype(tag);
            mtl::vec::dense_vector<double> x(n), b(n, 1.0);
            mtl::sparse::refine_options opt;
            opt.max_iter = 50;
            opt.rel_tol = 1e-14;
            return sf::supernodal_lu_solve_refined<float, Acc>(
                A, x, b, mtl::sparse::ordering::colamd{}, opt).rel_residual;
        };
        const double r1 = refine(float{});
        const double r2 = refine(fma_f{});
        const double r3 = refine(quire_f{});
        check(r3 == refine(exact_f{}), "[586:refine] the quire's residual is the Kulisch accumulator's");
        std::printf("  Hilbert(%zu)   plain %.1e   fma %.1e   quire %.1e\n", n, r1, r2, r3);
        check(r1 > 1e-6, "[586:refine] plain stalls");
        check(r2 > 1e-6, "[586:refine] fma stalls");
        check(r3 < 1e-10, "[586:refine] quire converges");
    }
}

} // namespace

int main() {
    direct_rows();
    refine_rows();
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_accumulation_matrix_direct: PASS\n";
    return 0;
}
