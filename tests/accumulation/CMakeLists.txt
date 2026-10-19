# The accumulation validation matrix, stillwater-sc/mtl5#586.

# The configuration-3 (quire) direct-solver and refinement rows (issue #11):
# dense and sparse factorizations and float-factor refinement, plain / FMA /
# quire at float. Configuration 1 is "the product rounds", so the compiler must
# not contract acc += a*b into an FMA.
mpir_add_test(test_accumulation_matrix_direct)
if(NOT MSVC)
	target_compile_options(test_accumulation_matrix_direct PRIVATE -ffp-contract=off)
endif()
