// mp-ir smoke test: verify the MTL5 + Universal composition builds and that LU
// iterative refinement (MTL5's mtl::lu_iterative_refine, driven here with
// Universal working/factor precisions) recovers accuracy -- factoring in a low
// Working precision while forming the residual in double drives the solution to
// an accuracy far better than the low-precision factor's own solve.
// Returns non-zero on failure (no external test framework, matching the repo's
// lightweight style).
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>

#include <sw/mp_ir/lu_iterative_refinement.hpp>

// Universal number types used as the low Working (factorization) precision.
#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

// A diagonally dominant system in double (the Residual/storage precision) with
// non-integer entries, so a low-precision factorization carries real rounding
// error that a double residual can then refine away.
void build_system(mtl::mat::dense2D<double>& A, mtl::vec::dense_vector<double>& b, std::size_t n) {
    A = mtl::mat::dense2D<double>(n, n);
    b = mtl::vec::dense_vector<double>(n, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (i == j) ? double(n) : 1.0 / double(i + j + 2);
}

// plain (unrefined) Working-precision LU residual ||b - A x||_inf, in double.
template <typename Working>
double plain_residual(const mtl::mat::dense2D<double>& A, const mtl::vec::dense_vector<double>& b) {
    const std::size_t n = A.num_rows();
    mtl::mat::dense2D<Working> LU(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) LU(i, j) = Working(A(i, j));
    std::vector<typename mtl::mat::dense2D<Working>::size_type> piv;
    mtl::lu_factor(LU, piv);
    mtl::vec::dense_vector<Working> bw(n, Working(0)), xw(n, Working(0));
    for (std::size_t i = 0; i < n; ++i) bw[i] = Working(b[i]);
    mtl::lu_solve(LU, piv, xw, bw);
    double r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t j = 0; j < n; ++j) ax += A(i, j) * double(xw[j]);
        r = std::max(r, std::abs(b[i] - ax));
    }
    return r;
}

template <typename Working>
bool ir_ok(const char* tag, double tol) {
    constexpr std::size_t n = 8;
    mtl::mat::dense2D<double> A;
    mtl::vec::dense_vector<double> b;
    build_system(A, b, n);

    mtl::vec::dense_vector<double> x;
    sw::mp_ir::ir_options opt;
    opt.max_iter = 30;
    auto res = sw::mp_ir::lu_iterative_refine<Working>(A, b, x, opt);

    const double plain = plain_residual<Working>(A, b);
    const bool ok = (res.rel_residual <= tol) && (res.rel_residual <= plain + 1e-300);
    if (!ok)
        std::cerr << tag << " LU-IR failed: refined=" << res.rel_residual
                  << " plain=" << plain << " tol=" << tol << '\n';
    else
        std::cout << tag << " LU-IR ok: refined=" << res.rel_residual
                  << " plain=" << plain << " (iters=" << res.iters << ")\n";
    return ok;
}

} // namespace

int main() {
    using namespace sw::universal;
    int failures = 0;

    // Working (factor) precision swept from full down to 16-bit; the residual is
    // always double, so refinement reaches the double floor for the wide factors
    // and a much-improved result for the narrow ones.
    if (!ir_ok<double>("double", 1e-12))            ++failures;
    if (!ir_ok<float>("float", 1e-10))              ++failures;
    if (!ir_ok<cfloat<16, 5>>("cfloat<16,5>", 1e-1)) ++failures;
    if (!ir_ok<posit<16, 2>>("posit<16,2>", 1e-1))   ++failures;

    if (failures == 0) std::cout << "mp-ir smoke test passed\n";
    return failures == 0 ? 0 : 1;
}
