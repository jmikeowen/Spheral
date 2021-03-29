include_directories(.)
set(Gravity_inst
  CompatibleGravitationalVelocityPolicy
  NBodyGravity
  )

set(Gravity_sources
  #LaplacianMatrixFactory.cc
  TreeGravityInst.cc
  )
instantiate(Gravity_inst Gravity_sources)

set(Gravity_headers
  CompatibleGravitationalVelocityPolicy.hh
  #LaplacianMatrixFactory.hh
  NBodyGravity.hh
  TreeGravity.hh
  TreeGravityInline.hh
  )


spheral_add_cxx_library(Gravity)

