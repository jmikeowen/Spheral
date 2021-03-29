#-----------------------------------------------------------------------------------
# spheral_add_cxx_library
#     - Generates the blt libraries for a given spheral C++ package
#
# package_name : *name* of spheral package to make into a library
#
# Variables that must be set before valling spheral_add_cxx_library:
#     ENABLE_STATIC_CXXONLY : Default False
#         - If set to true all libs will be made static
#     <package_Name>_headers
#         - List of headers to be installed
#     <package_name>_sources
#         - List of sources to build library with
#     spheral_depends
#         - List of targets the library depends on
#     spheral_blt_depends
#         - List of blt/libs the library depends on
#
#-----------------------------------------------------------------------------------

function(spheral_add_cxx_library package_name)

  # TODO : Check to see if -WL,--start-group ** -WL,--end-group is still needed or if it is a cmake version issue.
  if(ENABLE_STATIC_CXXONLY)
    # Build static spheral C++ library
    blt_add_library(NAME        Spheral_${package_name}
                    HEADERS     ${${package_name}_headers}
                    SOURCES     ${${package_name}_sources}
                    DEPENDS_ON  -Wl,--start-group ${spheral_blt_depends} ${${package_name}_ADDITIONAL_DEPENDS} -Wl,--end-group
                    SHARED      FALSE
                    )
  else()
    # Build shared spheral C++ library
    blt_add_library(NAME        Spheral_${package_name}
                    HEADERS     ${${package_name}_headers}
                    SOURCES     ${${package_name}_sources}
                    DEPENDS_ON  -Wl,--start-group ${spheral_blt_depends} ${${package_name}_ADDITIONAL_DEPENDS} -Wl,--end-group
                    SHARED      TRUE
                    )
  endif()

  # Only add target depends if the list exists, can throw an error otherwise
  if(spheral_depends)
    add_dependencies(Spheral_${package_name} ${spheral_depends})
  endif()


  # Install Spheral C++ target and set it as an exportable CMake target
  install(TARGETS             Spheral_${package_name}
          DESTINATION         lib
          EXPORT              ${PROJECT_NAME}-targets
          )

  # Install the headers
  install(FILES       ${${package_name}_headers}
          DESTINATION include/${package_name}
          )

  # Set the r-path of the C++ lib such that it is independent of the build dir when installed
  set_target_properties(Spheral_${package_name} PROPERTIES
                        INSTALL_RPATH           ${CMAKE_INSTALL_PREFIX}/lib
                        )

  # Add this to the SPHERAL_CXX_LIBS list
  get_property(SPHERAL_CXX_LIBS GLOBAL PROPERTY SPHERAL_CXX_LIBS)
  list(APPEND SPHERAL_CXX_LIBS Spheral_${package_name})
  set_property(GLOBAL PROPERTY SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}")

endfunction()


#-----------------------------------------------------------------------------------
# spheral_add_pybind11_library
#     - Generate the python friendly Spheral package lib
#
# package_name : *name* of spheral package to make into a library
#
# Variables that must be set before valling spheral_add_cxx_library:
#     <package_Name>_ADDITIONAL_INCLUDES
#         - List of addition includes needed by a given package
#     <package_name>_ADDITIONAL_SOURCE
#         - List of additional sources to build library with
#     SPHERAL_CXX_LIBS
#         - List of items that are required to build the python portion of spheral
#     spheral_depends
#         - List of targets the library depends on
#     spheral_blt_depends
#         - List of blt/libs the library depends on
#
#-----------------------------------------------------------------------------------

function(spheral_add_pybind11_library package_name)
  include(${CMAKE_MODULE_PATH}/spheral/PYB11Generator.cmake)

  get_property(SPHERAL_CXX_LIBS GLOBAL PROPERTY SPHERAL_CXX_LIBS)

  # Generate the pybind11 C++ source file
  PYB11_GENERATE_BINDINGS(${package_name})

  # TODO : Check to see if -WL,--start-group ** -WL,--end-group is still needed or if it is a cmake version issue.
  # Build python friendly spheral lib
  set(MODULE_NAME Spheral${package_name})
  blt_add_library(NAME         ${MODULE_NAME}
                  SOURCES      ${PYB11_GENERATED_SOURCE} ${${package_name}_ADDITIONAL_SOURCES}
                  DEPENDS_ON   -Wl,--start-group ${SPHERAL_CXX_LIBS} ${spheral_blt_depends} ${${package_name}_ADDITIONAL_DEPENDS} -Wl,--end-group
                  INCLUDES     ${${package_name}_ADDITIONAL_INCLUDES}
                  OUTPUT_NAME  ${MODULE_NAME}
                  CLEAR_PREFIX TRUE
                  SHARED       TRUE
                  )
  add_dependencies(${MODULE_NAME} ${spheral_py_depends} ${spheral_depends})
  target_compile_options(${MODULE_NAME} PRIVATE
                         "-Wno-unused-local-typedefs"
                         "-Wno-self-assign-overloaded"
                         "-Wno-overloaded-virtual"
                         "-Wno-delete-non-abstract-non-virtual-dtor")

  install(TARGETS     ${MODULE_NAME}
          DESTINATION Spheral
          )

  # Set the r-path of the C++ lib such that it is independent of the build dir when installed
  set_target_properties(${MODULE_NAME} PROPERTIES
                        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${boost_DIR}/lib;${python_DIR}/lib"
                        )

endfunction()
