//---------------------------------Spheral++----------------------------------//
// StrengthModel -- The interface base class for strength models.
//
// Created by JMO, Wed Sep 8 15:18:44 2004
//----------------------------------------------------------------------------//
#include "StrengthModel.hh"
#include "Field/Field.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template<typename Dimension>
StrengthModel<Dimension>::
StrengthModel() {
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
StrengthModel<Dimension>::
~StrengthModel() {
}

//------------------------------------------------------------------------------
// Sound speed
//------------------------------------------------------------------------------
template<typename Dimension>
void
StrengthModel<Dimension>::
soundSpeed(Field<Dimension, Scalar>& soundSpeed,
           const Field<Dimension, Scalar>& /*density*/,
           const Field<Dimension, Scalar>& /*specificThermalEnergy*/,
           const Field<Dimension, Scalar>& /*pressure*/,
           const Field<Dimension, Scalar>& fluidSoundSpeed,
           const Field<Dimension, SymTensor>& /*damage*/) const {
  soundSpeed = fluidSoundSpeed;
}

//------------------------------------------------------------------------------
// Bulk modulus
//------------------------------------------------------------------------------
template<typename Dimension>
void
StrengthModel<Dimension>::
bulkModulus(Field<Dimension, Scalar>& /*bulkModulus*/,
            const Field<Dimension, Scalar>& /*density*/,
            const Field<Dimension, Scalar>& /*specificThermalEnergy*/) const {
  VERIFY2(false,
          "StrengthModel::bulkModulus called");
}

//------------------------------------------------------------------------------
// Melt energy
//------------------------------------------------------------------------------
template<typename Dimension>
void
StrengthModel<Dimension>::
meltSpecificEnergy(Field<Dimension, Scalar>& meltSpecificEnergy,
                   const Field<Dimension, Scalar>& /*density*/,
                   const Field<Dimension, Scalar>& /*specficThermalEnergy*/) const {
  meltSpecificEnergy = 0.0;
}

//------------------------------------------------------------------------------
// Cold energy
//------------------------------------------------------------------------------
template<typename Dimension>
void
StrengthModel<Dimension>::
coldSpecificEnergy(Field<Dimension, Scalar>& coldSpecificEnergy,
                   const Field<Dimension, Scalar>& /*density*/,
                   const Field<Dimension, Scalar>& /*specficThermalEnergy*/) const {
  coldSpecificEnergy = 0.0;
}

}
