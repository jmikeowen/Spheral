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
    weibullFlawDistributionBenzAsphaug.hh
    weibullFlawDistributionOwen.hh
    YoungsModulusPolicy.hh
    )

spheral_install_python_files(
  identifyFragments.py
  WeibullDamage.py
  GradyKippTensorDamage.py
  JohnsonCookDamageFactories.py
  )

spheral_add_cxx_library(Damage)

