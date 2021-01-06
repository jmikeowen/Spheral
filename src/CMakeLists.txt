################################
# Build spheral packages 
################################
include(${CMAKE_MODULE_PATH}/spheral/InstantiateCXX.cmake)
include(${CMAKE_MODULE_PATH}/spheral/SpheralAddLibs.cmake)
include(${CMAKE_MODULE_PATH}/spheral/SpheralInstallPythonFiles.cmake)

include_directories(.)

set(SPHERAL_PYTHON_INSTALL ${PROJECT_BINARY_DIR}/lib)

list(APPEND _packages 
  ArtificialViscosity
  ArtificialConduction
  Boundary
  CRKSPH
  Damage
  DataBase
  DataOutput
  Distributed
  ExternalForce
  Field
  FieldOperations
  FileIO
  Geometry
  Gravity
  Hydro
  Integrator
  Kernel
  Material
  Mesh
  Neighbor
  NodeGenerators
  NodeList
  Physics
  RK
  SolidMaterial
  SPH
  Strength
  SVPH
  Utilities
  )

if(NOT ENABLE_CXXONLY)
  list(APPEND _packages 
    PythonMPIInterfaces
    SimulationControl
    Pybind11Wraps
    )
endif()

if(NOT ENABLE_MPI)
  list(REMOVE_ITEM _packages
    Distributed
    )
endif()

foreach(_package ${_packages})
  add_subdirectory(${_package})
endforeach()

include_directories(${extra_packages_DIR})
foreach(e_package ${extra_packages})
  add_subdirectory(${extra_packages_DIR}/${e_package} ${CMAKE_CURRENT_BINARY_DIR}/${e_package})
endforeach()

#add_subdirectory(CXXTests)

install(EXPORT ${PROJECT_NAME}-targets DESTINATION Spheral/lib/cmake)
set(SPHERAL_CXX_LIBS ${SPHERAL_CXX_LIBS} PARENT_SCOPE)
