include_directories(.)
set(Damage_inst
    DamageModel
    TensorDamageModel
    StrainPolicy
    TensorStrainPolicy
    TensorDamagePolicy
    YoungsModulusPolicy
    LongitudinalSoundSpeedPolicy
    weibullFlawDistribution
    computeFragmentField
    JohnsonCookDamage
    JohnsonCookFailureStrainPolicy
    JohnsonCookDamagePolicy
   )


set(Damage_sources )

instantiate(Damage_inst Damage_sources)

set(Damage_headers
    computeFragmentField.hh
    DamagedSoundSpeedPolicy.hh
    DamageGradientPolicy.hh
    DamageModel.hh
    DamageModelInline.hh
    EffectiveTensorDamagePolicy.hh
    GradyKippScalarDamage.hh
    JohnsonCookDamage.hh
    JohnsonCookDamageInline.hh
    JohnsonCookDamagePolicy.hh
    JohnsonCookFailureStrainPolicy.hh
    LongitudinalSoundSpeedPolicy.hh
    oneMinusDamage.hh
    ScalarDamageModel.hh
    ScalarDamagePolicy.hh
    StrainPolicy.hh
    TensorDamageModel.hh
    TensorDamageModelInline.hh
    TensorDamagePolicy.hh
    TensorStrainPolicy.hh
    weibullFlawDistribution.hh
    YoungsModulusPolicy.hh
    )

spheral_install_python_files(
  identifyFragments.py
  WeibullDamage.py
  GradyKippTensorDamage.py
  JohnsonCookDamageFactories.py
  )

spheral_add_cxx_library(Damage)

list(APPEND SPHERAL_CXX_LIBS Spheral_Damage)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
