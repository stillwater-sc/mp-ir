// round_and_replace: LU-IR with round-and-replace squeeze across low precisions.
// Migrated from Universal applications/performance/ir/roundAndReplace.cpp
// (author James Quinlan), re-expressed on MTL5's LU-IR core.
#include "../scaling_study.hpp"
int main(int argc, char** argv) {
    using namespace sw::mp_ir::study;
    std::string matrix = (argc > 1) ? argv[1] : "lehmer";
    std::size_t n      = (argc > 2) ? std::stoul(argv[2]) : 8;
    double scale       = (argc > 3) ? std::stod(argv[3]) : 1.0e5;   // large -> stresses the low precision
    run("round-and-replace", squeeze_kind::round_and_replace, matrix, n, scale, High(1));
    return 0;
}
