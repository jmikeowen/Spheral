include_directories(.)
set(Gravity_inst
  CompatibleGravitationalVelocityPolicy
  NBodyGravity
  )

set(Gravity_sources
  TreeGravityInst.cc
  )
instantiate(Gravity_inst Gravity_sources)

set(Gravity_headers
  CompatibleGravitationalVelocityPolicy.hh
  NBodyGravity.hh
  TreeGravity.hh
  TreeGravityInline.hh
  )


spheral_add_cxx_library(Gravity)

