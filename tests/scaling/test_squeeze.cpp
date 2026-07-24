// mp-ir test: the squeeze preconditioners (round-and-replace / scale-and-round /
// two-sided). Each conditions a high-precision matrix into a low precision's
// range; because the study derives b from the squeezed A, the exact solution is
// unchanged, so a low-precision LU-IR must still converge to the all-ones vector.
// Returns non-zero on failure (no external framework).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/generators/lehmer.hpp>

#include <sw/mp_ir/squeeze.hpp>
#include <sw/mp_ir/lu_iterative_refinement.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

using High = double;   // converts to/from both posit and cfloat low precisions

mtl::mat::dense2D<High> lehmer_scaled(std::size_t n, double scale) {
    mtl::generators::lehmer<double> g(n);
    mtl::mat::dense2D<High> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = scale * g(i, j);
    return A;
}

// LU-IR forward error ||X - x||_inf for a High system with a known all-ones X.
template <typename Low>
double solve_fe(const mtl::mat::dense2D<High>& A) {
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<High> b(n, High(0)), x;
    for (std::size_t i = 0; i < n; ++i) { High s(0); for (std::size_t j = 0; j < n; ++j) s = s + A(i, j); b[i] = s; }
    sw::mp_ir::ir_options opt; opt.max_iter = 20;
    sw::mp_ir::lu_iterative_refine<Low>(A, b, x, opt);
    double fe = 0.0;
    for (std::size_t i = 0; i < n; ++i) fe = std::max(fe, std::abs(double(x[i]) - 1.0));
    return fe;
}

double max_abs(const mtl::mat::dense2D<High>& A) {
    double m = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) m = std::max(m, std::abs(double(A(i, j))));
    return m;
}

} // namespace

int main() {
    using namespace sw::universal;
    using Pos = posit<16, 2>;
    using Fp8 = cfloat<8, 4, std::uint8_t, true, false, false>;   // overflow-prone low precision
    int failures = 0;
    const std::size_t n = 6;

    // scale_and_round: max|A| becomes ~T; the squeezed system still solves to ones
    {
        auto A = lehmer_scaled(n, 1.0e4);
        High mu = sw::mp_ir::scale_and_round<Pos>(A, 0.5);
        if (!(std::abs(max_abs(A) - 0.5) <= 1e-3)) { std::cerr << "scale_and_round: max|A|=" << max_abs(A) << " != 0.5\n"; ++failures; }
        if (mu <= 0.0) { std::cerr << "scale_and_round: mu <= 0\n"; ++failures; }
        double fe = solve_fe<Pos>(A);
        if (fe > 1e-6) { std::cerr << "scale_and_round: IR fe=" << fe << "\n"; ++failures; }
    }

    // round_and_replace: an fp8-overflowing matrix is clamped under fp8 maxpos
    {
        auto A = lehmer_scaled(n, 1.0e4);
        const double fp8max = double(Fp8(SpecificValue::maxpos));
        if (!(max_abs(A) > fp8max)) { std::cerr << "round_and_replace: test setup did not overflow fp8\n"; ++failures; }
        sw::mp_ir::round_and_replace<Fp8>(A);
        if (max_abs(A) > fp8max * (1.0 + 1e-9)) { std::cerr << "round_and_replace: max|A|=" << max_abs(A) << " > fp8 maxpos=" << fp8max << "\n"; ++failures; }
    }

    // two_sided: row+col equilibration then scale; squeezed system solves to ones
    {
        auto A = lehmer_scaled(n, 1.0e4);
        mtl::vec::dense_vector<High> R, S;
        sw::mp_ir::two_sided_scale_and_round<Pos>(A, 0.5, R, S);
        double fe = solve_fe<Pos>(A);
        if (fe > 1e-6) { std::cerr << "two_sided: IR fe=" << fe << "\n"; ++failures; }
    }

    if (failures == 0) std::cout << "mp-ir squeeze test passed\n";
    return failures == 0 ? 0 : 1;
}
