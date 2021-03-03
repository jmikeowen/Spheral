include(ExternalProject)

#-------------------------------------------------------------------------------
# Configure CMake
#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

if (NOT CMAKE_MODULE_PATH)
  set(CMAKE_MODULE_PATH "${SPHERAL_ROOT_DIR}/cmake")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS On)

#-------------------------------------------------------------------------------
# Optionally suppress compiler warnings
#-------------------------------------------------------------------------------
option(ENABLE_WARNINGS "show compiler warnings" ON)
if (NOT ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()
message("-- compiler warnings ${ENABLE_WARNINGS}")

option(ENABLE_UNUSED_VARIABLE_WARNINGS "show unused variable compiler warnings" ON)
if (NOT ENABLE_UNUSED_VARIABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
endif()
message("-- compiler unused variable warnings ${ENABLE_UNUSED_VARIABLE_WARNINGS}")

#-------------------------------------------------------------------------------
# Configure and Include blt
#-------------------------------------------------------------------------------
set(ENABLE_MPI ON CACHE BOOL "")
set(ENABLE_OPENMP ON CACHE BOOL "")

if(NOT SPHERAL_BLT_DIR) 
  set (SPHERAL_BLT_REL_DIR "${SPHERAL_ROOT_DIR}/cmake/blt" CACHE PATH "")
  get_filename_component(SPHERAL_BLT_DIR "${SPHERAL_BLT_REL_DIR}" ABSOLUTE)
endif()

if (NOT EXISTS "${SPHERAL_BLT_DIR}/SetupBLT.cmake")
    message(FATAL_ERROR 
            "${SPHERAL_BLT_DIR} is not present.\n"
            "call cmake with -DSPHERAL_BLT_DIR=/your/installation/of/blt\n")
endif()

include(${SPHERAL_BLT_DIR}/SetupBLT.cmake)

#-------------------------------------------------------------------------------
# Include standard build system logic and options / definitions
#-------------------------------------------------------------------------------
set(ENABLE_CXXONLY OFF CACHE BOOL "enable C++ only build without python bindings")
set(ENABLE_1D ON CACHE BOOL "enable 1d")
set(ENABLE_2D ON CACHE BOOL "enable 2d")
set(ENABLE_3D ON CACHE BOOL "enable 3d")
set(ENABLE_INSTANTIATIONS ON CACHE BOOL "enable instantiations")
set(ENABLE_TIMER OFF CACHE BOOL "enable timer")
set(ENABLE_ANEOS ON CACHE BOOL "enable the ANEOS equation of state package")
set(ENABLE_OPENSUBDIV ON CACHE BOOL "enable the Opensubdiv Pixar extension for refining polyhedra")
set(ENABLE_HELMHOLTZ ON CACHE BOOL "enable the Helmholtz equation of state package")

option(ENABLE_STATIC_CXXONLY "build only static libs" OFF)
if(ENABLE_STATIC_CXXONLY)
  set(ENABLE_CXXONLY ON)
endif()

if(ENABLE_MPI)
  set(BLT_MPI_COMPILE_FLAGS -DUSE_MPI -DMPICH_SKIP_MPICXX -ULAM_WANT_MPI2CPP -DOMPI_SKIP_MPICXX)
  list(APPEND spheral_blt_depends mpi)
endif()

if(ENABLE_OPENMP)
  list(APPEND spheral_blt_depends openmp)
endif()

option(BOOST_HEADER_ONLY "only use the header only components of Boost" OFF)

#-------------------------------------------------------------------------------#
# Set a default build type if none was specified
#-------------------------------------------------------------------------------#
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build (debug, release, etc)." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#-------------------------------------------------------------------------------
# Should we build sphinx documentation
#-------------------------------------------------------------------------------
set(ENABLE_DOCS OFF CACHE BOOL "enable sphinx Spheral documentation")

#-------------------------------------------------------------------------------
# Install / Locate third party libraries
#-------------------------------------------------------------------------------
set(SPHERAL_TPL_DIR "" CACHE STRING "Directory to install Spheral TPLs and/or Spheral libs.")
if (CMAKE_INSTALL_PREFIX)
  if (SPHERAL_TPL_DIR STREQUAL "")
    set(SPHERAL_TPL_DIR ${CMAKE_INSTALL_PREFIX}/tpl)
    message("-- Setting SPHERAL_TPL_DIR ${SPHERAL_TPL_DIR}")
  endif()
endif()

include(${SPHERAL_ROOT_DIR}/cmake/InstallTPLs.cmake)

include(${SPHERAL_ROOT_DIR}/cmake/CMakeDefinitions.cmake)

#-------------------------------------------------------------------------------
# Set full rpath information by default
#-------------------------------------------------------------------------------
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#-------------------------------------------------------------------------------
# We need the set of Spheral C++ libraries globally
#-------------------------------------------------------------------------------
set_property(GLOBAL PROPERTY SPHERAL_CXX_LIBS)

if (NOT BUILD_TPL_ONLY)
  #-------------------------------------------------------------------------------
  # Install symlink for spheral->python
  #-------------------------------------------------------------------------------
  if (NOT ENABLE_CXXONLY)
    install(CODE "execute_process( \
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${PYTHON_EXE} spheral \
      WORKING_DIRECTORY ${SPHERAL_TPL_DIR})")
  endif()

  #-------------------------------------------------------------------------------
  # Prepare to build the src
  #-------------------------------------------------------------------------------
  add_subdirectory(${SPHERAL_ROOT_DIR}/src)

  #-------------------------------------------------------------------------------
  # Add the documentation
  #-------------------------------------------------------------------------------
  if (NOT ENABLE_CXXONLY)
    add_subdirectory(${SPHERAL_ROOT_DIR}/docs)
  endif()
endif()
