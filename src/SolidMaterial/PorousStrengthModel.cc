//---------------------------------Spheral++----------------------------------//
// PorousStrengthModel
// 
// See header for references and such.
//----------------------------------------------------------------------------//

#include "PorousStrengthModel.hh"
#include "Field/Field.hh"

namespace Spheral {

using std::min;
using std::max;
using std::abs;

//------------------------------------------------------------------------------
// Constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
PorousStrengthModel<Dimension>::
PorousStrengthModel(const StrengthModel<Dimension>& solidStrength):
  StrengthModel<Dimension>(),
  mSolidStrength(solidStrength),
  mAlphaPtr(0) {
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
PorousStrengthModel<Dimension>::
~PorousStrengthModel() {
}

//------------------------------------------------------------------------------
// Set the shear modulus.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PorousStrengthModel<Dimension>::
shearModulus(Field<Dimension, Scalar>& shearModulus,
             const Field<Dimension, Scalar>& density,
             const Field<Dimension, Scalar>& specificThermalEnergy,
             const Field<Dimension, Scalar>& pressure,
             const Field<Dimension, SymTensor>& damage) const {
  REQUIRE(density.nodeListPtr() == shearModulus.nodeListPtr());
  REQUIRE(specificThermalEnergy.nodeListPtr() == shearModulus.nodeListPtr());
  REQUIRE(pressure.nodeListPtr() == shearModulus.nodeListPtr());
  REQUIRE(damage.nodeListPtr() == shearModulus.nodeListPtr());
  REQUIRE(mAlphaPtr->nodeListPtr() == shearModulus.nodeListPtr());

  // The base model sets the solid (compacted) value.
  const Field<Dimension, Scalar> rhoS = (*mAlphaPtr)*density;
  mSolidStrength.shearModulus(shearModulus, rhoS, specificThermalEnergy, pressure, damage);

  // Now apply the porosity modifier.
  const unsigned n = shearModulus.numInternalElements();
  for (unsigned i = 0; i != n; ++i) {
    shearModulus(i) /= (*mAlphaPtr)(i);
  }
}

//------------------------------------------------------------------------------
// Set the yield strength.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PorousStrengthModel<Dimension>::
yieldStrength(Field<Dimension, Scalar>& yieldStrength,
              const Field<Dimension, Scalar>& density,
              const Field<Dimension, Scalar>& specificThermalEnergy,
              const Field<Dimension, Scalar>& pressure,
              const Field<Dimension, Scalar>& plasticStrain,
              const Field<Dimension, Scalar>& plasticStrainRate,
              const Field<Dimension, SymTensor>& damage) const {
  REQUIRE(density.nodeListPtr() == yieldStrength.nodeListPtr());
  REQUIRE(specificThermalEnergy.nodeListPtr() == yieldStrength.nodeListPtr());
  REQUIRE(pressure.nodeListPtr() == yieldStrength.nodeListPtr());
  REQUIRE(damage.nodeListPtr() == yieldStrength.nodeListPtr());
  REQUIRE(plasticStrain.nodeListPtr() == yieldStrength.nodeListPtr());
  REQUIRE(plasticStrainRate.nodeListPtr() == yieldStrength.nodeListPtr());
  REQUIRE(mAlphaPtr->nodeListPtr() == yieldStrength.nodeListPtr());

  // The base model sets the solid (compacted) value.
  const Field<Dimension, Scalar> rhoS = (*mAlphaPtr)*density;
  mSolidStrength.yieldStrength(yieldStrength, rhoS, specificThermalEnergy, pressure, plasticStrain, plasticStrainRate, damage);

  // Now apply the porosity modifier.
  const unsigned n = yieldStrength.numInternalElements();
  for (unsigned i = 0; i != n; ++i) {
    yieldStrength(i) /= (*mAlphaPtr)(i);
  }
}

//------------------------------------------------------------------------------
// Sound speed
//------------------------------------------------------------------------------
template<typename Dimension>
void
PorousStrengthModel<Dimension>::
soundSpeed(Field<Dimension, Scalar>& soundSpeed,
           const Field<Dimension, Scalar>& density,
           const Field<Dimension, Scalar>& specificThermalEnergy,
           const Field<Dimension, Scalar>& pressure,
           const Field<Dimension, Scalar>& fluidSoundSpeed,
           const Field<Dimension, SymTensor>& damage) const {
  mSolidStrength.soundSpeed(soundSpeed, density, specificThermalEnergy, pressure, fluidSoundSpeed, damage);
}

//------------------------------------------------------------------------------
// Bulk modulus
//------------------------------------------------------------------------------
template<typename Dimension>
void
PorousStrengthModel<Dimension>::
bulkModulus(Field<Dimension, Scalar>& bulkModulus,
            const Field<Dimension, Scalar>& density,
            const Field<Dimension, Scalar>& specificThermalEnergy) const {
  mSolidStrength.bulkModulus(bulkModulus, density, specificThermalEnergy);
}

//------------------------------------------------------------------------------
// Melt energy
//------------------------------------------------------------------------------
template<typename Dimension>
void
PorousStrengthModel<Dimension>::
meltSpecificEnergy(Field<Dimension, Scalar>& meltSpecificEnergy,
                   const Field<Dimension, Scalar>& density,
                   const Field<Dimension, Scalar>& specificThermalEnergy) const {
  mSolidStrength.meltSpecificEnergy(meltSpecificEnergy, density, specificThermalEnergy);
}

//------------------------------------------------------------------------------
// Cold energy
//------------------------------------------------------------------------------
template<typename Dimension>
void
PorousStrengthModel<Dimension>::
coldSpecificEnergy(Field<Dimension, Scalar>& coldSpecificEnergy,
                   const Field<Dimension, Scalar>& density,
                   const Field<Dimension, Scalar>& specificThermalEnergy) const {
  mSolidStrength.coldSpecificEnergy(coldSpecificEnergy, density, specificThermalEnergy);
}

//------------------------------------------------------------------------------
// Access the underlying solid strength model.
//------------------------------------------------------------------------------
template<typename Dimension>
const StrengthModel<Dimension>&
PorousStrengthModel<Dimension>::
solidStrength() const {
  return mSolidStrength;
}

//------------------------------------------------------------------------------
// Access the alpha field.
//------------------------------------------------------------------------------
template<typename Dimension>
const Field<Dimension, typename Dimension::Scalar>&
PorousStrengthModel<Dimension>::
alpha() const {
  return *mAlphaPtr;
}

template<typename Dimension>
void
PorousStrengthModel<Dimension>::
alpha(const Field<Dimension, typename Dimension::Scalar>& x) {
  mAlphaPtr = &x;
}

}
