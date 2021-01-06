#------------------------------------------------------------------------------
# Specify list of tests
#------------------------------------------------------------------------------

set(gtest_spheral_tests
    basic.cc
   )


# set(spheral_test_depends
#     spheral_blt_depends
#    )

#------------------------------------------------------------------------------
# Add gtest C++ tests
#------------------------------------------------------------------------------
foreach(test ${gtest_spheral_tests})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}_test
                        SOURCES ${test}
                        OUTPUT_DIR ${${PROJECT_BINARY_DIR}/tests}
                        DEPENDS_ON ${SPHERAL_CXX_LIBS} ${spheral_blt_depends} gtest
                        )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name}_test
                  )
endforeach()
