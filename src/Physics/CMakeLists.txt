include_directories(.)
set(Physics_inst
    GenericBodyForce
    GenericHydro
    Physics
   )

set(Physics_sources )

instantiate(Physics_inst Physics_sources)

set(Physics_headers
    GenericBodyForce.hh
    GenericHydro.hh
    GenericHydroInline.hh
    Physics.hh
    PhysicsInline.hh
    )

spheral_add_cxx_library(Physics)

