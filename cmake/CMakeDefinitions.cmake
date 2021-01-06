#-----------------------------------------------------------------------------------
# Definitions to be added as compile flags for spheral 
#-----------------------------------------------------------------------------------

# If we're building debug default DBC to All
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("-- building Debug")
  add_definitions("-DDEBUG=1")
  if (NOT DBC_MODE)
    set(DBC_MODE "All")
  endif()
else()
  add_definitions("-DDEBUG=0")
endif()

# The DBC flag
if (DBC_MODE STREQUAL "All")
  message("-- DBC (design by contract) set to All")
  add_definitions("-DDBC_COMPILE_ALL")
elseif (DBC_MODE STREQUAL "Pre")
  message("-- DBC (design by contract) set to Pre")
  add_definitions("-DDBC_COMPILE_PRE")
else()
  message("-- DBC (design by contract) off")
endif()

# Bound checking option -- very expensive at run time
if (ENABLE_BOUNDCHECKING)
  message("-- bound checking enabled")
  add_definitions(-D_GLIBCXX_DEBUG=1)
else()
  message("-- bound checking disabled")
endif()

# CXXONLY
if (ENABLE_CXXONLY)
  add_definitions(-DCXXONLY=1)
endif()

# Default Polytope options
add_definitions(-DUSE_TETGEN=0)
add_definitions(-DUSE_TRIANGLE=0)
add_definitions(-DNOPOLYTOPE=1)

# Are we using Opensubdiv?
if (ENABLE_OPENSUBDIV)
  add_definitions(-DENABLE_OPENSUBDIV)
endif()

# Choose the dimensions we build
if (ENABLE_1D)
  add_definitions(-DSPHERAL1D=1)
endif()
if (ENABLE_2D)
  add_definitions(-DSPHERAL2D=1)
endif()
if (ENABLE_3D)
  add_definitions(-DNOR3D=1)
  add_definitions(-DSPHERAL3D=1)
endif()

if (ENABLE_TIMER)
  add_definitions(-DTIMER=1)
endif()

if (ENABLE_MPI)
    add_definitions(-DUSE_MPI=1)
endif()
