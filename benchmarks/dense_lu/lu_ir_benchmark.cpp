// lu_ir_benchmark: cost/accuracy of LU iterative refinement.
//
// For each working precision, times the working-precision LU factorization and
// the refinement sweeps, and reports the refinement steps needed to reach the
// residual floor. The point of mixed-precision IR is that the expensive factor
// runs in low precision while a cheap higher-precision residual recovers
// accuracy -- this benchmark is where that trade-off gets measured as the IR
// experiments migrate in from Universal (see docs/roadmap.md).
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

// deterministic, diagonally dominant system (no <random> so the benchmark is
// reproducible and free of Date/rand dependencies).
template <typename T>
void build_system(mtl::mat::dense2D<T>& A, mtl::vec::dense_vector<T>& b, std::size_t n) {
    A = mtl::mat::dense2D<T>(n, n);
    b = mtl::vec::dense_vector<T>(n, T(0));
    for (std::size_t i = 0; i < n; ++i) {
        T s(0);
        for (std::size_t j = 0; j < n; ++j) {
            std::size_t d = (i > j) ? (i - j) : (j - i);
            T v = (i == j) ? T(2 * n) : (d <= 2 ? T(1) / T(int(d) + 1) : T(0));
            A(i, j) = v;
            s = s + v;
        }
        b[i] = s;                        // exact solution is all ones
    }
}

template <typename Clock>
double ms_since(const typename Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <typename T>
void bench(const std::string& name, std::size_t n) {
    using Clock = std::chrono::steady_clock;
    mtl::mat::dense2D<T> A;
    mtl::vec::dense_vector<T> b;
    build_system(A, b, n);

    // time the working-precision factorization
    auto t0 = Clock::now();
    mtl::mat::dense2D<T> LU(A);
    std::vector<typename mtl::mat::dense2D<T>::size_type> piv;
    mtl::lu_factor(LU, piv);
    double factor_ms = ms_since<Clock>(t0);

    // time the full refined solve
    mtl::vec::dense_vector<T> x;
    sw::mp_ir::ir_options opt;
    opt.max_iterations = 30;
    auto t1 = Clock::now();
    auto res = sw::mp_ir::lu_ir_solve<double>(A, b, x, opt);
    double solve_ms = ms_since<Clock>(t1);

    std::cout << std::left << std::setw(16) << name
              << "  factor=" << std::setw(9) << std::fixed << std::setprecision(3) << factor_ms << " ms"
              << "  refine=" << std::setw(9) << solve_ms << " ms"
              << "  iters=" << std::setw(3) << res.iterations
              << "  residual=" << std::scientific << std::setprecision(2) << res.final_residual << '\n';
}

} // namespace

int main(int argc, char** argv) {
    using namespace sw::universal;
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::stoul(argv[1])) : 64;

    std::cout << "LU iterative-refinement benchmark, n = " << n << " (residual precision: double)\n";
    std::cout << "working precision       factor time      refine time     steps  residual\n";
    bench<double>("double", n);
    bench<float>("float", n);
    bench<cfloat<16, 5>>("cfloat<16,5>", n);
    bench<posit<16, 2>>("posit<16,2>", n);
    bench<posit<32, 2>>("posit<32,2>", n);
    return 0;
}
