# Applications (demonstration programs)
# Organized by iterative-refinement category (dense_lu, scaling, sparse); each
# leaf subdirectory contains its own CMakeLists.txt.

# LU iterative refinement: refined-vs-unrefined accuracy across working
# precisions with a double-precision residual.
add_subdirectory(dense_lu/lu_ir_precision)

# LUIR: three-precision LU iterative refinement (migrated from Universal
# applications/performance/ir, author James Quinlan).
add_subdirectory(dense_lu/luir)
