// lu_ir_benchmark: cost/accuracy of LU iterative refinement across factorization
// precisions. The system is stored in double; the LU factorization runs in a low
// Working precision, and a double residual refines the solution (MTL5's
// mtl::lu_iterative_refine driven with Universal working precisions).
//
// The point of mixed-precision IR is that the expensive factor runs in low
// precision while a cheap higher-precision residual recovers accuracy -- this
// benchmark measures that trade-off (factor vs refine time, steps to floor) as
// the IR experiments migrate in from Universal (see docs/roadmap.md).
#include <chrono>
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

// deterministic, diagonally dominant, non-integer system (reproducible; no
// <random> so there is no Date/rand dependency).
void build_system(mtl::mat::dense2D<double>& A, mtl::vec::dense_vector<double>& b, std::size_t n) {
    A = mtl::mat::dense2D<double>(n, n);
    b = mtl::vec::dense_vector<double>(n, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (i == j) ? double(2 * n) : 1.0 / double(i + j + 2);
}

template <typename Clock>
double ms_since(const typename Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <typename Working>
void bench(const std::string& name, std::size_t n) {
    using Clock = std::chrono::steady_clock;
    mtl::mat::dense2D<double> A;
    mtl::vec::dense_vector<double> b;
    build_system(A, b, n);

    // time the low-precision factorization in isolation
    auto t0 = Clock::now();
    mtl::mat::dense2D<Working> LU(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) LU(i, j) = Working(A(i, j));
    std::vector<typename mtl::mat::dense2D<Working>::size_type> piv;
    mtl::lu_factor(LU, piv);
    double factor_ms = ms_since<Clock>(t0);

    // time the full refined solve
    mtl::vec::dense_vector<double> x;
    sw::mp_ir::ir_options opt;
    opt.max_iter = 30;
    auto t1 = Clock::now();
    auto res = sw::mp_ir::lu_iterative_refine<Working>(A, b, x, opt);
    double solve_ms = ms_since<Clock>(t1);

    std::cout << std::left << std::setw(16) << name
              << "  factor=" << std::setw(9) << std::fixed << std::setprecision(3) << factor_ms << " ms"
              << "  refine=" << std::setw(9) << solve_ms << " ms"
              << "  iters=" << std::setw(3) << res.iters
              << "  residual=" << std::scientific << std::setprecision(2) << res.rel_residual << '\n';
}

} // namespace

int main(int argc, char** argv) {
    using namespace sw::universal;
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::stoul(argv[1])) : 64;

    std::cout << "LU iterative-refinement benchmark, n = " << n << " (storage/residual: double)\n";
    std::cout << "factor precision        factor time      refine time     steps  residual\n";
    bench<double>("double", n);
    bench<float>("float", n);
    bench<cfloat<16, 5>>("cfloat<16,5>", n);
    bench<posit<16, 2>>("posit<16,2>", n);
    bench<posit<32, 2>>("posit<32,2>", n);
    return 0;
}
