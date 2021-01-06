include_directories(.)
set(Strength_inst
    BulkModulusPolicy
    DeviatoricStressPolicy
    MeltEnergyPolicy
    PlasticStrainPolicy
    ShearModulusPolicy
    StrengthSoundSpeedPolicy
    YieldStrengthPolicy
   )

set(Strength_sources
    SolidFieldNames.cc
    )

if(ENABLE_2D)
  list(APPEND Strength_sources
       #RZDeviatoricStressPolicy.cc
       RZPlasticStrainPolicy.cc
      )
endif()

instantiate(Strength_inst Strength_sources)

set(Strength_headers
    BulkModulusPolicy.hh
    DeviatoricStressPolicy.hh
    MeltEnergyPolicy.hh
    PlasticStrainPolicy.hh
    #RZDeviatoricStressPolicy.hh
    RZPlasticStrainPolicy.hh
    ShearModulusPolicy.hh
    SolidFieldNames.hh
    StrengthSoundSpeedPolicy.hh
    YieldStrengthPolicy.hh
    effectiveKernelVolume.hh
    )

spheral_add_cxx_library(Strength)

list(APPEND SPHERAL_CXX_LIBS Spheral_Strength)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
