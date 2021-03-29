include_directories(.)
set(RK_inst
  computeRKSumVolume
  computeOccupancyVolume
  computeHullVolumes
  computeHVolumes
  HVolumePolicy
  ContinuityVolumePolicy
  computeRKVolumes
  RKCorrections
  RKUtilities012
  RKUtilities3
  RKUtilities4
  RKUtilities5
  RKUtilities6
  RKUtilities7
  ReproducingKernelMethods
  ReproducingKernel
  interpolateRK
  gradientRK
  hessianRK
  )

set(RK_sources
  computeVoronoiVolume.cc
  computeVoronoiVolume1d.cc
  RKFieldNames.cc
  )

if(ENABLE_2D)
  list(APPEND RK_sources
    ContinuityVolumePolicyRZ.cc
    )
endif()

instantiate(RK_inst RK_sources)

set(RK_headers
  computeHullVolumes.hh
  computeHVolumes.hh
  computeOccupancyVolume.hh
  computeRKSumVolume.hh
  computeRKVolumes.hh
  computeVoronoiVolume.hh
  ContinuityVolumePolicy.hh
  ContinuityVolumePolicyRZ.hh
  HullVolumePolicy.hh
  HVolumePolicy.hh
  interpolateRK.hh
  ReproducingKernel.hh
  ReproducingKernelInline.hh
  ReproducingKernelMethods.hh
  ReproducingKernelMethodsInline.hh
  RKCoefficients.hh
  RKCorrectionParams.hh
  RKCorrections.hh
  RKUtilities.hh
  RKUtilitiesInline.hh
  )

spheral_add_cxx_library(RK)

