include_directories(.)
set(Material_inst
    EquationOfState
    GammaLawGas
    PolytropicEquationOfState
    IsothermalEquationOfState
   )

set(Material_sources
    PhysicalConstants.cc
    )

set(Material_headers
    EquationOfState.hh
    EquationOfStateInline.hh
    PhysicalConstants.hh
    PhysicalConstantsInline.hh
    )

spheral_install_python_files(
  MaterialUnits.py
  MaterialEquationsOfState.py
  )

# Helmholtz
if (ENABLE_HELMHOLTZ)
  message("-- Helmholtz equation of state enabled")
  list(APPEND Material_inst      HelmholtzEquationOfState)
  list(APPEND Material_headers   HelmholtzEquationOfState.hh)
  list(APPEND Material_sources   public_helm.f90 invert_helm.f90)
else()
  message("-- Helmholtz equation of state disabled")
endif()

instantiate(Material_inst Material_sources)

spheral_add_cxx_library(Material)

list(APPEND SPHERAL_CXX_LIBS Spheral_Material)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
