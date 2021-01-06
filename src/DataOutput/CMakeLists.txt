include_directories(.)
set(DataOutput_sources
    RestartRegistrar.cc
    #RestartableObject.cc
    )

set(DataOutput_headers
    Restart.hh
    RestartHandle.hh
    RestartInline.hh
    RestartRegistrar.hh
    RestartRegistrarInline.hh
    registerWithRestart.hh
    )

spheral_add_cxx_library(DataOutput)

list(APPEND SPHERAL_CXX_LIBS Spheral_DataOutput)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
