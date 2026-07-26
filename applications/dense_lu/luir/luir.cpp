// luir: A = LU iterative refinement.
//
// Migrated from the Universal Mixed-Precision Iterative Refinement project
// (applications/performance/ir/luir.cpp, original author James Quinlan) onto
// MTL5's Universal-free dense LU-IR core (mtl::lu_iterative_refine) driven with
// Universal posit precisions.
//
// Three-precision mixed-precision iterative refinement (Higham, "Accuracy and
// Stability of Numerical Algorithms", ch. 12): factor A in a low precision, then
// recover accuracy by refining with a residual formed in a high precision. Here
//   Low  = posit<16,5>   (the cheap factorization precision)
//   High = posit<64,5>   (storage + residual precision)
// matching the original configs.hpp; the original's intermediate Working
// precision (posit<32,8>, where the LU was stored) is subsumed by MTL5's
// two-precision core (factor precision + residual precision).
//
// Test matrices come from MTL5's Universal-free layer: the parametric generators
// (minij/hilbert/lehmer/frank/moler/pascal) for arbitrary size, and the named
// test-matrix catalog (mtl::testsuite -- the migrated SuiteSparse/textbook
// problems: bcsstk*, west*, steam*, fs_183*, saylr1, gre_343, ...) for the same
// fixed problems the Universal originals used (universal#1210).
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/spectral_properties.hpp>     // condition_number
#include <mtl/generators/minij.hpp>
#include <mtl/generators/hilbert.hpp>
#include <mtl/generators/lehmer.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/pascal.hpp>
#include <mtl/generators/testsuite.hpp>   // named SuiteSparse/textbook catalog (universal#1210)

#include <sw/mp_ir/lu_iterative_refinement.hpp>

#include <universal/number/posit/posit.hpp>

namespace {

using Low  = sw::universal::posit<16, 5>;   // low (factorization) precision
using High = sw::universal::posit<64, 5>;   // high (storage + residual) precision

// Materialize a named test matrix as a double reference. Implicit generators
// (minij/hilbert/lehmer) are read element-wise; dense factories
// (frank/moler/pascal) are returned directly. Default "dd" is a well-conditioned
// diagonally dominant system.
mtl::mat::dense2D<double> make_reference(const std::string& name, std::size_t n) {
    // Named catalog matrices (SuiteSparse/textbook, e.g. bcsstk01, west0132,
    // steam1) are fixed-size problems -- n is ignored for these.
    if (mtl::testsuite::kappa_table().count(name)) return mtl::testsuite::by_name(name);

    mtl::mat::dense2D<double> A(n, n);
    auto fill = [&](auto gen) { for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) A(i, j) = gen(i, j); };
    if      (name == "minij")   fill(mtl::generators::minij<double>(n));
    else if (name == "hilbert") fill(mtl::generators::hilbert<double>(n));
    else if (name == "lehmer")  fill(mtl::generators::lehmer<double>(n));
    else if (name == "frank")   A = mtl::generators::frank<double>(n);
    else if (name == "moler")   A = mtl::generators::moler<double>(n);
    else if (name == "pascal")  A = mtl::generators::pascal<double>(n);
    else                        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j)
                                    A(i, j) = (i == j) ? double(n) : 1.0 / double(i + j + 2);
    return A;
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace sw::universal;

    const std::string matrix = (argc > 1) ? argv[1] : "minij";
    std::size_t       n      = (argc > 2) ? static_cast<std::size_t>(std::stoul(argv[2])) : 8;
    const int         maxit  = (argc > 3) ? std::atoi(argv[3]) : 10;

    // build the reference in double, then cast to the High storage precision.
    // Named catalog matrices are fixed-size, so take n from the actual matrix
    // (for the parametric generators this already equals the requested n).
    mtl::mat::dense2D<double> Ad = make_reference(matrix, n);
    n = Ad.num_rows();

    // catalog matrices carry a published condition number; generators are
    // estimated from the assembled matrix.
    const double cond = mtl::testsuite::kappa_table().count(matrix)
                        ? mtl::testsuite::kappa(matrix)
                        : mtl::condition_number(Ad);

    mtl::mat::dense2D<High> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = High(Ad(i, j));

    // known solution X = [1,...,1]; right-hand side b = A X (High precision)
    mtl::vec::dense_vector<High> b(n, High(0));
    for (std::size_t i = 0; i < n; ++i) { High s(0); for (std::size_t j = 0; j < n; ++j) s = s + A(i, j); b[i] = s; }

    // solve by LU iterative refinement: factor in Low, refine with a High residual
    mtl::vec::dense_vector<High> x;
    sw::mp_ir::ir_options opt;
    opt.max_iter = maxit;
    auto res = sw::mp_ir::lu_iterative_refine<Low>(A, b, x, opt);

    // forward error ||X - x||_inf and normwise backward error
    double fe = 0.0;
    for (std::size_t i = 0; i < n; ++i) fe = std::max(fe, std::abs(double(x[i]) - 1.0));
    const double nbe   = sw::mp_ir::normwise_backward_error(A, x, b);
    const double u_W   = double(std::numeric_limits<posit<32, 8>>::epsilon());  // working-precision roundoff
    const bool   converged = nbe < u_W;

    std::cout << std::setprecision(7);
    std::cout << "LU iterative refinement  (Low = " << type_tag(Low()) << ", High = " << type_tag(High()) << ")\n";
    std::cout << "matrix           : " << matrix << "  (" << n << " x " << n << ")\n";
    std::cout << "condition number : " << std::scientific << cond << '\n';
    std::cout << "iterations       : " << res.iters << (converged ? "   (converged)" : "   (did not reach working precision)") << '\n';
    std::cout << "forward error    : " << fe << "   ||X - x||_inf\n";
    std::cout << "backward error   : " << nbe << "   (working roundoff u_W = " << u_W << ")\n";
    std::cout << "rel residual     : " << res.rel_residual << '\n';

    std::cout << "\n   x (approx)              x (exact)\n";
    std::cout << "   ----------------------------------\n";
    const std::size_t z = (n < 10) ? n : 10;
    for (std::size_t i = 0; i < z; ++i)
        std::cout << "   " << std::setw(20) << std::left << double(x[i]) << "    " << 1.0 << '\n';

    return EXIT_SUCCESS;
}
catch (const std::exception& err) {
    std::cerr << "luir: caught exception: " << err.what() << std::endl;
    return EXIT_FAILURE;
}
