include_directories(.)
set(ArtificialConduction_inst
    ArtificialConduction
    ArtificialConductionPolicy
    )

set(ArtificialConduction_sources )
instantiate(ArtificialConduction_inst ArtificialConduction_sources)

set(ArtificialConduction_headers
    ArtificialConduction.hh
    ArtificialConductionPolicy.hh
    ) 

spheral_add_cxx_library(ArtificialConduction)

list(APPEND SPHERAL_CXX_LIBS Spheral_ArtificialConduction)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
