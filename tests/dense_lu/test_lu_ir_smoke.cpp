// mp-ir smoke test: verify the MTL5 + Universal composition builds and that LU
// iterative refinement solves a small dense system in each number type -- and
// that refining with a double-precision residual drives the solution to an
// accuracy far better than the working precision's own (unrefined) LU solve.
// Returns non-zero on failure (no external test framework, matching the repo's
// lightweight style).
#include <cmath>
#include <cstddef>
#include <iostream>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>

#include <sw/mp_ir/lu_iterative_refinement.hpp>

// Universal number types used as the working precision of the factorization.
#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

// A small, well-conditioned, diagonally dominant system whose exact solution
// is the all-ones vector (b = A * ones).
template <typename T>
void build_system(mtl::mat::dense2D<T>& A, mtl::vec::dense_vector<T>& b) {
    constexpr std::size_t n = 4;
    const T vals[n][n] = {
        {T(4), T(1), T(0), T(1)},
        {T(1), T(5), T(2), T(0)},
        {T(0), T(2), T(6), T(1)},
        {T(1), T(0), T(1), T(4)},
    };
    A = mtl::mat::dense2D<T>(n, n);
    b = mtl::vec::dense_vector<T>(n, T(0));
    for (std::size_t i = 0; i < n; ++i) {
        T s(0);
        for (std::size_t j = 0; j < n; ++j) { A(i, j) = vals[i][j]; s = s + vals[i][j]; }
        b[i] = s;                       // row sum -> exact solution is all ones
    }
}

// ||b - A x||_inf measured in double (precision-independent yardstick).
template <typename T>
double residual_inf(const mtl::mat::dense2D<T>& A, const mtl::vec::dense_vector<T>& b,
                    const mtl::vec::dense_vector<T>& x) {
    const std::size_t n = A.num_rows();
    double r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t j = 0; j < n; ++j) ax += double(A(i, j)) * double(x[j]);
        r = std::max(r, std::abs(double(b[i]) - ax));
    }
    return r;
}

template <typename T>
bool lu_ir_ok(const char* tag, double tol) {
    mtl::mat::dense2D<T> A;
    mtl::vec::dense_vector<T> b;
    build_system(A, b);

    // refined solve (residual formed in double)
    mtl::vec::dense_vector<T> x;
    sw::mp_ir::ir_options opt;
    opt.max_iterations = 15;
    auto res = sw::mp_ir::lu_ir_solve<double>(A, b, x, opt);
    double refined = res.final_residual;

    // plain (unrefined) working-precision LU solve, for comparison
    mtl::mat::dense2D<T> LU(A);
    std::vector<typename mtl::mat::dense2D<T>::size_type> piv;
    mtl::lu_factor(LU, piv);
    mtl::vec::dense_vector<T> x0(A.num_rows(), T(0));
    mtl::lu_solve(LU, piv, x0, b);
    double plain = residual_inf(A, b, x0);

    const bool ok = (refined <= tol) && (refined <= plain + 1e-300);
    if (!ok) {
        std::cerr << tag << " LU-IR failed: refined=" << refined
                  << " plain=" << plain << " tol=" << tol << '\n';
    }
    else {
        std::cout << tag << " LU-IR ok: refined=" << refined
                  << " plain=" << plain << " (iters=" << res.iterations << ")\n";
    }
    return ok;
}

} // namespace

int main() {
    using namespace sw::universal;
    int failures = 0;

    if (!lu_ir_ok<double>("double", 1e-12))          ++failures;
    if (!lu_ir_ok<float>("float", 1e-6))             ++failures;
    if (!lu_ir_ok<cfloat<16, 5>>("cfloat<16,5>", 1e-2)) ++failures;
    if (!lu_ir_ok<posit<16, 2>>("posit<16,2>", 1e-2))   ++failures;

    if (failures == 0) std::cout << "mp-ir smoke test passed\n";
    return failures == 0 ? 0 : 1;
}
