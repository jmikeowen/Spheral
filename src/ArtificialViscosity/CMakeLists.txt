include_directories(.)
set(ArtificialViscosity_inst
    ArtificialViscosity
    CRKSPHMonaghanGingoldViscosity
    CullenDehnenViscosity
    FiniteVolumeViscosity
    IncrementCullenMultipliers
    MonaghanGingoldViscosity
    MorrisMonaghanReducingViscosity
    TensorSVPHViscosity
    TensorCRKSPHViscosity
    TensorMonaghanGingoldViscosity
    VonNeumanViscosity
   )


set(ArtificialViscosity_sources )

if(ENABLE_2D)
  list(APPEND ArtificialViscosity_sources
       MonaghanGingoldViscosityGSRZ.cc
      )
endif()

instantiate(ArtificialViscosity_inst ArtificialViscosity_sources)

set(ArtificialViscosity_headers
    ArtificialViscosity.hh
    ArtificialViscosityInline.hh
    ArtificialViscosityList.hh
    CRKSPHMonaghanGingoldViscosity.hh
    CRKSPHMonaghanGingoldViscosityRZ.hh
    CullenDehnenViscosity.hh
    FiniteVolumeViscosity.hh
    MonaghanGingoldViscosity.hh
    MonaghanGingoldViscosityGSRZ.hh
    MonaghanGingoldViscosityRZ.hh
    TensorCRKSPHViscosity.hh
    TensorMonaghanGingoldViscosity.hh
    )

spheral_add_cxx_library(ArtificialViscosity)

list(APPEND SPHERAL_CXX_LIBS Spheral_ArtificialViscosity)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
