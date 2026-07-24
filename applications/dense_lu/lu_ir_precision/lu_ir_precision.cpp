// lu_ir_precision: LU iterative refinement across factorization precisions.
//
// The system is stored in double; the LU factorization is computed in a low
// Working precision, then a double-precision residual refines the solution
// (MTL5's mtl::lu_iterative_refine, driven with Universal working precisions).
// Prints, per Working precision, the residual of the plain (unrefined) solve vs
// the refined solve -- showing how much accuracy the higher-precision residual
// recovers from a low-precision factor.
//
// This is the demonstration analogue of Universal's applications/performance/ir
// studies; the round-and-replace / scale-and-round variants migrate here from
// Universal in a follow-up (see docs/roadmap.md).
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>

#include <sw/mp_ir/lu_iterative_refinement.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

// diagonally dominant, non-integer entries (so a low-precision factor is inexact)
void build_system(mtl::mat::dense2D<double>& A, mtl::vec::dense_vector<double>& b, std::size_t n) {
    A = mtl::mat::dense2D<double>(n, n);
    b = mtl::vec::dense_vector<double>(n, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (i == j) ? double(n) : 1.0 / double(i + j + 2);
}

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
void report(const std::string& name, std::size_t n) {
    mtl::mat::dense2D<double> A;
    mtl::vec::dense_vector<double> b;
    build_system(A, b, n);

    double plain = plain_residual<Working>(A, b);

    mtl::vec::dense_vector<double> x;
    sw::mp_ir::ir_options opt;
    opt.max_iter = 30;
    auto res = sw::mp_ir::lu_iterative_refine<Working>(A, b, x, opt);

    std::cout << std::left << std::setw(16) << name
              << "  plain=" << std::setw(12) << std::scientific << std::setprecision(3) << plain
              << "  refined=" << std::setw(12) << res.rel_residual
              << "  iters=" << res.iters << '\n';
}

} // namespace

int main(int argc, char** argv) {
    using namespace sw::universal;
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::stoul(argv[1])) : 8;

    std::cout << "LU iterative refinement (storage: double, residual: double), n = " << n << "\n";
    std::cout << "factor precision        plain (unrefined)   refined\n";
    report<double>("double", n);
    report<float>("float", n);
    report<cfloat<16, 5>>("cfloat<16,5>", n);
    report<posit<16, 2>>("posit<16,2>", n);
    report<posit<32, 2>>("posit<32,2>", n);
    return 0;
}
