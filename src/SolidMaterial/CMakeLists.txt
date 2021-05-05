include_directories(.)
set(SolidMaterial_inst
    LinearPolynomialEquationOfState
    GruneisenEquationOfState
    OsborneEquationOfState
    MurnahanEquationOfState
    TillotsonEquationOfState
    StrengthModel
    ConstantStrength
    SteinbergGuinanStrength
    JohnsonCookStrength
    CollinsStrength
    PorousEquationOfState
    PorousStrengthModel
    StrainPorosity
    PhysicsEvolvingMaterialLibrary
    )

set(SolidMaterial_sources )

set(SolidMaterial_headers
    ANEOS.hh
    CollinsStrength.hh
    ConstantStrength.hh
    Geodyn.hh
    GruneisenEquationOfState.hh
    GruneisenEquationOfStateInline.hh
    JohnsonCookStrength.hh
    LinearPolynomialEquationOfState.hh
    LinearPolynomialEquationOfStateInline.hh
    MurnahanEquationOfState.hh
    MurnahanEquationOfStateInline.hh
    NullStrength.hh
    OsborneEquationOfState.hh
    OsborneEquationOfStateInline.hh
    PhysicsEvolvingMaterialLibrary.hh
    PolynomialFit.hh
    PolynomialFitInline.hh
    PorousEquationOfState.hh
    PorousStrengthModel.hh
    SolidEquationOfState.hh
    SolidEquationOfStateInline.hh
    SteinbergGuinanLundStrength.hh
    SteinbergGuinanStrength.hh
    StrainPorosity.hh
    StrainPorosityInline.hh
    StrengthModel.hh
    TillotsonEquationOfState.hh
    TillotsonEquationOfStateInline.hh
    )

spheral_install_python_files(
    MaterialPropertiesLib.py
    SolidMaterialUnits.py
    SolidMaterialEquationsOfState.py
    ShadowTillotsonEquationOfState.py
    ShadowGruneisenEquationOfState.py
    ShadowConstantStrength.py
    )

# ANEOS
if (ENABLE_ANEOS)
  message("-- ANEOS enabled")
  list(APPEND SolidMaterial_inst      ANEOS)
  list(APPEND SolidMaterial_headers   ANEOS.hh)
  list(APPEND SolidMaterial_sources   ANEOS_initialize.f)
else()
  message("-- ANEOS disabled")
endif()

instantiate(SolidMaterial_inst SolidMaterial_sources)

# Ignore -W-maybe-uninitialized warnings thrown from bisectSearch for possibly uninitialized iterators from boost multi-array.
if (NOT ENABLE_CUDA)
  set_source_files_properties(ANEOSInst1d.cc ANEOSInst2d.cc ANEOSInst3d.cc PROPERTIES COMPILE_FLAGS -Wno-uninitialized)
endif()

spheral_add_cxx_library(SolidMaterial)

