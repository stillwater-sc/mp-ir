include(CTest)

# Helper: each test is a self-checking executable linked against sw::mp_ir.
function(mpir_add_test test_name)
	add_executable(${test_name} ${test_name}.cpp)
	target_link_libraries(${test_name} PRIVATE sw::mp_ir)
	add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Tests are organized by iterative-refinement category:
#   1. dense_lu  -- LU-based iterative refinement (factor low, refine high)
#   2. scaling   -- squeeze / scale-and-round strategies (round-and-replace,
#                   one-/two-sided scaling into a low-precision factor)
#   3. sparse    -- sparse direct factor + refinement (drives MTL5's
#                   Universal-free sparse iterative_refine core)
add_subdirectory(dense_lu)
add_subdirectory(scaling)
add_subdirectory(sparse)
