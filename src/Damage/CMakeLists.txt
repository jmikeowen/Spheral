include_directories(.)
set(Damage_inst
    DamageModel
    TensorDamageModel
    StrainPolicy
    TensorStrainPolicy
    TensorDamagePolicy
    YoungsModulusPolicy
    LongitudinalSoundSpeedPolicy
    weibullFlawDistributionBenzAsphaug
    weibullFlawDistributionOwen
    computeFragmentField
    JohnsonCookDamage
    JohnsonCookFailureStrainPolicy
    JohnsonCookDamagePolicy
    DamagedPressurePolicy
   )


set(Damage_sources )

instantiate(Damage_inst Damage_sources)

set(Damage_headers
    DamageGradientPolicy.hh
    DamageModel.hh
    DamageModelInline.hh
    DamagedPressurePolicy.hh
    DamagedSoundSpeedPolicy.hh
    EffectiveTensorDamagePolicy.hh
    GradyKippScalarDamage.hh
    JohnsonCookDamage.hh
    JohnsonCookDamageInline.hh
    JohnsonCookDamagePolicy.hh
    JohnsonCookFailureStrainPolicy.hh
    LongitudinalSoundSpeedPolicy.hh
    ScalarDamageModel.hh
    ScalarDamagePolicy.hh
    StrainPolicy.hh
    TensorDamageModel.hh
    TensorDamageModelInline.hh
    TensorDamagePolicy.hh
    TensorStrainPolicy.hh
    YoungsModulusPolicy.hh
    computeFragmentField.hh
    oneMinusDamage.hh
    weibullFlawDistributionBenzAsphaug.hh
    weibullFlawDistributionOwen.hh
    )

spheral_install_python_files(
  identifyFragments.py
  WeibullDamage.py
  GradyKippTensorDamage.py
  JohnsonCookDamageFactories.py
  )

spheral_add_cxx_library(Damage)

