// scale_and_round: LU-IR with scale-and-round squeeze across low precisions.
// Migrated from Universal applications/performance/ir/scaleAndRound.cpp
// (author James Quinlan), re-expressed on MTL5's LU-IR core.
#include "../scaling_study.hpp"
int main(int argc, char** argv) {
    using namespace sw::mp_ir::study;
    std::string matrix = (argc > 1) ? argv[1] : "minij";
    std::size_t n      = (argc > 2) ? std::stoul(argv[2]) : 8;
    double scale       = (argc > 3) ? std::stod(argv[3]) : 1.0e5;
    double T           = (argc > 4) ? std::stod(argv[4]) : 0.5;   // target magnitude for max|A|
    run("scale-and-round", squeeze_kind::scale_and_round, matrix, n, scale, High(T));
    return 0;
}
