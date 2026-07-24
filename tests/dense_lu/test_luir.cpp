// mp-ir test: the migrated LUIR experiment. Three-precision (posit) LU iterative
// refinement -- factor in Low = posit<16,5>, refine with a High = posit<64,5>
// residual -- converges to the known all-ones solution on well-conditioned
// matrices, and improves on the plain low-precision solve. Returns non-zero on
// failure (no external framework, matching the repo's lightweight style).
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/generators/minij.hpp>
#include <mtl/generators/lehmer.hpp>

#include <sw/mp_ir/lu_iterative_refinement.hpp>

#include <universal/number/posit/posit.hpp>

namespace {

using Low  = sw::universal::posit<16, 5>;
using High = sw::universal::posit<64, 5>;

template <typename Gen>
mtl::mat::dense2D<High> build(Gen gen, std::size_t n) {
    mtl::mat::dense2D<High> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = High(gen(i, j));
    return A;
}

double plain_low_forward_error(const mtl::mat::dense2D<High>& A, const mtl::vec::dense_vector<High>& b, std::size_t n) {
    // plain Low-precision LU solve, forward error vs the all-ones exact solution
    mtl::mat::dense2D<Low> LU(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) LU(i, j) = Low(double(A(i, j)));
    std::vector<typename mtl::mat::dense2D<Low>::size_type> piv;
    mtl::lu_factor(LU, piv);
    mtl::vec::dense_vector<Low> bl(n, Low(0)), xl(n, Low(0));
    for (std::size_t i = 0; i < n; ++i) bl[i] = Low(double(b[i]));
    mtl::lu_solve(LU, piv, xl, bl);
    double fe = 0.0;
    for (std::size_t i = 0; i < n; ++i) fe = std::max(fe, std::abs(double(xl[i]) - 1.0));
    return fe;
}

template <typename Gen>
bool luir_ok(const char* tag, Gen gen, std::size_t n, double tol) {
    auto A = build(gen, n);
    mtl::vec::dense_vector<High> b(n, High(0));
    for (std::size_t i = 0; i < n; ++i) { High s(0); for (std::size_t j = 0; j < n; ++j) s = s + A(i, j); b[i] = s; }

    mtl::vec::dense_vector<High> x;
    sw::mp_ir::ir_options opt; opt.max_iter = 10;
    auto res = sw::mp_ir::lu_iterative_refine<Low>(A, b, x, opt);

    double fe = 0.0;
    for (std::size_t i = 0; i < n; ++i) fe = std::max(fe, std::abs(double(x[i]) - 1.0));
    double plain = plain_low_forward_error(A, b, n);

    const bool ok = (fe <= tol) && (fe <= plain + 1e-300);
    if (!ok) std::cerr << tag << " LUIR failed: refined fe=" << fe << " plain fe=" << plain << " tol=" << tol << '\n';
    else     std::cout << tag << " LUIR ok: refined fe=" << fe << " plain fe=" << plain << " iters=" << res.iters << '\n';
    return ok;
}

} // namespace

int main() {
    int failures = 0;
    if (!luir_ok("minij(8)",  mtl::generators::minij<double>(8),  8,  1e-8)) ++failures;
    if (!luir_ok("lehmer(8)", mtl::generators::lehmer<double>(8), 8,  1e-6)) ++failures;
    if (failures == 0) std::cout << "mp-ir luir test passed\n";
    return failures == 0 ? 0 : 1;
}
