#-----------------------------------------------------------------------------------
# Define the list of Third Party Libs to be installed here
#-----------------------------------------------------------------------------------

# Do NOT add any TPLs to the clean target
set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

# Initialize TPL options
include(${SPHERAL_ROOT_DIR}/cmake/spheral/SpheralHandleTPL.cmake)

# If set to Off NONE of the TPLs will be built and installed
# it is expected that the user provide locations for each one
# else the default install location will be searched for TPLs
set(BUILD_TPL ON CACHE BOOL "Define if any TPLs will be built or not.")

# Default build flags for each TPL
set(zlib_BUILD ON CACHE BOOL "Option to build zlib")
set(boost_BUILD ON CACHE BOOL "Option to build boost")
set(eigen_BUILD ON CACHE BOOL "Option to build eigen")
set(qhull_BUILD ON CACHE BOOL "Option to build qhull")
set(polytope_BUILD ON CACHE BOOL "Option to build polytope")
set(hdf5_BUILD ON CACHE BOOL "Option to build hdf5")
set(silo_BUILD ON CACHE BOOL "Option to build silo")
set(opensubdiv_BUILD ON CACHE BOOL "Option to build Opensubdiv")
set(aneos_BUILD ON CACHE BOOL "Option to build ANEOS third party lib")
set(conduit_BUILD ON CACHE BOOL "Option to build Conduit")
set(axom_BUILD ON CACHE BOOL "Option to build Axom")
set(raja_BUILD ON CACHE BOOL "Option to build RAJA")
set(polyclipper_BUILD ON CACHE BOOL "Option to build PolyClipper")

set(pybind11_BUILD ON CACHE BOOL "Option to build pybind11")
set(python_BUILD ON CACHE BOOL "Option to build python")
set(pip_BUILD ON CACHE BOOL "Option to build pip")

# These libs are always needed
Spheral_Handle_TPL(zlib spheral_depends)
Spheral_Handle_TPL(boost spheral_depends)
Spheral_Handle_TPL(eigen spheral_depends)
Spheral_Handle_TPL(qhull spheral_depends)
Spheral_Handle_TPL(hdf5 spheral_depends)
Spheral_Handle_TPL(silo spheral_depends)
Spheral_Handle_TPL(conduit spheral_depends)
Spheral_Handle_TPL(axom spheral_depends)
Spheral_Handle_TPL(raja spheral_depends)

# Some libraries are optional
if (ENABLE_ANEOS)
  Spheral_Handle_TPL(aneos spheral_depends)
endif()
if (ENABLE_OPENSUBDIV)
  Spheral_Handle_TPL(opensubdiv spheral_depends)
endif()

# Only needed when building the python interface of spheral
if(NOT ENABLE_CXXONLY)
  Spheral_Handle_TPL(python spheral_depends)
  Spheral_Handle_TPL(pip spheral_py_depends)
  include(${SPHERAL_ROOT_DIR}/cmake/tpl/pythonModule.cmake)
  Spheral_Handle_TPL(pybind11 spheral_depends)
endif()

Spheral_Handle_TPL(polytope spheral_depends)
Spheral_Handle_TPL(polyclipper spheral_depends)

if (EXISTS ${EXTERNAL_SPHERAL_TPL_CMAKE})
  include(${EXTERNAL_SPHERAL_TPL_CMAKE})
endif()
