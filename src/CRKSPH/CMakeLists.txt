include_directories(.)
set(CRKSPH_inst
    computeCRKSPHSumMassDensity
    computeSolidCRKSPHSumMassDensity
    CRKSPHHydroBase
    detectSurface
    SolidCRKSPHHydroBase
    editMultimaterialSurfaceTopology
    zerothOrderSurfaceCorrections
   )


set(CRKSPH_sources
    centerOfMass.cc
   )

if(ENABLE_2D)
  list(APPEND CRKSPH_sources
       CRKSPHHydroBaseRZ.cc
       SolidCRKSPHHydroBaseRZ.cc
      )
endif()

if(ENABLE_3D)
  list(APPEND CRKSPH_sources
      )
endif()

instantiate(CRKSPH_inst CRKSPH_sources)

set(CRKSPH_headers
    CRKSPHHydroBase.hh
    CRKSPHHydroBaseInline.hh
    CRKSPHHydroBaseRZ.hh
    SolidCRKSPHHydroBase.hh
    SolidCRKSPHHydroBaseInline.hh
    SolidCRKSPHHydroBaseRZ.hh
    centerOfMass.hh
    computeCRKSPHEvaluation.hh
    computeCRKSPHIntegral.hh
    computeCRKSPHSumMassDensity.hh
    computeHullSumMassDensity.hh
    computeSolidCRKSPHSumMassDensity.hh
    computeVoronoiCentroids.hh
    detectSurface.hh
    volumeSpacing.hh
    zerothOrderSurfaceCorrections.hh
    )

spheral_add_cxx_library(CRKSPH)
    
spheral_install_python_files(CRKSPHHydros.py)

list(APPEND SPHERAL_CXX_LIBS Spheral_CRKSPH)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
