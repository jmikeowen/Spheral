include_directories(.)
set(SVPH_inst
    computeSVPHCorrections
    computeSVPHCorrectionsOnFaces
    computeSumVoronoiCellMassDensityFromFaces
    sampleFieldListSVPH
    gradientFieldListSVPH
    SVPHFacetedHydroBase
    SVPHCorrectionsPolicy
    SVPHMassDensityPolicy
    CompatibleFaceSpecificThermalEnergyPolicy
    MeshIdealHPolicy
    SpecificThermalEnergyVolumePolicy
    CellPressurePolicy
   )


set(SVPH_sources 
  SVPHFieldNames.cc
  )

instantiate(SVPH_inst SVPH_sources)

set(SVPH_headers
    CellPressurePolicy.hh
    CompatibleFaceSpecificThermalEnergyPolicy.hh
    computeSumVoronoiCellMassDensityFromFaces.hh
    computeSVPHCorrections.hh
    computeSVPHCorrectionsOnFaces.hh
    gradientFieldListSVPH.hh
    MeshIdealHPolicy.hh
    sampleFieldListSVPH.hh
    SpecificThermalEnergyVolumePolicy.hh
    SVPHCorrectionsPolicy.hh
    SVPHFacetedHydroBase.hh
    SVPHFacetedHydroBaseInline.hh
    SVPHHydroBase.hh
    SVPHHydroBaseInline.hh
    SVPHMassDensityPolicy.hh
    )

spheral_install_python_files(SVPHHydros.py)

spheral_add_cxx_library(SVPH)

list(APPEND SPHERAL_CXX_LIBS Spheral_SVPH)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
