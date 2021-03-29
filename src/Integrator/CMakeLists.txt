include_directories(.)
set(Integrator_inst
    Integrator
    PredictorCorrector 
    SynchronousRK1
    SynchronousRK2
    SynchronousRK4
    CheapSynchronousRK2
    Verlet
    )

set(Integrator_sources )

instantiate(Integrator_inst Integrator_sources)

set(Integrator_headers
    CheapSynchronousRK2.hh
    Integrator.hh
    IntegratorInline.hh
    )

spheral_add_cxx_library(Integrator)

