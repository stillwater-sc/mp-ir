// lu_ir_precision: LU iterative refinement across working precisions.
//
// Factors a small diagonally dominant system in a low working precision, then
// refines with a double-precision residual. Prints, per number type, the
// residual of the plain (unrefined) working-precision LU solve vs the refined
// solve -- showing how much accuracy a higher-precision residual recovers.
//
// This is the demonstration analogue of Universal's applications/performance/ir
// studies; the full round-and-replace / scale-and-round variants migrate here
// from Universal in a follow-up (see docs/roadmap.md).
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

template <typename T>
void build_system(mtl::mat::dense2D<T>& A, mtl::vec::dense_vector<T>& b, std::size_t n) {
    A = mtl::mat::dense2D<T>(n, n);
    b = mtl::vec::dense_vector<T>(n, T(0));
    for (std::size_t i = 0; i < n; ++i) {
        T s(0);
        for (std::size_t j = 0; j < n; ++j) {
            // diagonally dominant tridiagonal-ish: diag 4, off-diag -1
            T v = (i == j) ? T(4) : ((i + 1 == j || j + 1 == i) ? T(-1) : T(0));
            A(i, j) = v;
            s = s + v;
        }
        b[i] = s;                        // exact solution is all ones
    }
}

template <typename T>
double plain_residual(const mtl::mat::dense2D<T>& A, const mtl::vec::dense_vector<T>& b) {
    mtl::mat::dense2D<T> LU(A);
    std::vector<typename mtl::mat::dense2D<T>::size_type> piv;
    mtl::lu_factor(LU, piv);
    mtl::vec::dense_vector<T> x(A.num_rows(), T(0));
    mtl::lu_solve(LU, piv, x, b);
    double r = 0.0;
    const std::size_t n = A.num_rows();
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t j = 0; j < n; ++j) ax += double(A(i, j)) * double(x[j]);
        r = std::max(r, std::abs(double(b[i]) - ax));
    }
    return r;
}

template <typename T>
void report(const std::string& name, std::size_t n) {
    mtl::mat::dense2D<T> A;
    mtl::vec::dense_vector<T> b;
    build_system(A, b, n);

    double plain = plain_residual<T>(A, b);

    mtl::vec::dense_vector<T> x;
    sw::mp_ir::ir_options opt;
    opt.max_iterations = 20;
    auto res = sw::mp_ir::lu_ir_solve<double>(A, b, x, opt);

    std::cout << std::left << std::setw(16) << name
              << "  plain=" << std::setw(12) << std::scientific << std::setprecision(3) << plain
              << "  refined=" << std::setw(12) << res.final_residual
              << "  iters=" << res.iterations << '\n';
}

} // namespace

int main(int argc, char** argv) {
    using namespace sw::universal;
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::stoul(argv[1])) : 8;

    std::cout << "LU iterative refinement -- residual ||b - A x||_inf (double), n = " << n << "\n";
    std::cout << "working precision       plain (unrefined)   refined (double residual)\n";
    report<double>("double", n);
    report<float>("float", n);
    report<cfloat<16, 5>>("cfloat<16,5>", n);
    report<posit<16, 2>>("posit<16,2>", n);
    report<posit<32, 2>>("posit<32,2>", n);
    return 0;
}
