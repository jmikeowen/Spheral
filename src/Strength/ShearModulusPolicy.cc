//---------------------------------Spheral++----------------------------------//
// ShearModulusPolicy -- An implementation of UpdatePolicyBase specialized
// for the updating the dependent shear modulus state.
//
// Created by JMO, Thu Sep 30 23:01:00 PDT 2004
//----------------------------------------------------------------------------//
#include "ShearModulusPolicy.hh"
#include "NodeList/SolidNodeList.hh"
#include "SolidMaterial/StrengthModel.hh"
#include "SolidFieldNames.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Strength/SolidFieldNames.hh"
#include "DataBase/UpdatePolicyBase.hh"
#include "DataBase/IncrementState.hh"
#include "DataBase/ReplaceState.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "Field/FieldList.hh"
#include "Utilities/DBC.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
ShearModulusPolicy<Dimension>::
ShearModulusPolicy():
  FieldListUpdatePolicyBase<Dimension, typename Dimension::Scalar>(HydroFieldNames::massDensity,
                                                                   HydroFieldNames::specificThermalEnergy,
                                                                   HydroFieldNames::pressure,
                                                                   SolidFieldNames::tensorDamage) {
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
ShearModulusPolicy<Dimension>::
~ShearModulusPolicy() {
}

//------------------------------------------------------------------------------
// Update the field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
ShearModulusPolicy<Dimension>::
update(const KeyType& key,
       State<Dimension>& state,
       StateDerivatives<Dimension>& /*derivs*/,
       const double /*multiplier*/,
       const double /*t*/,
       const double /*dt*/) {
  KeyType fieldKey, nodeListKey;
  StateBase<Dimension>::splitFieldKey(key, fieldKey, nodeListKey);
  REQUIRE(fieldKey == SolidFieldNames::shearModulus and 
          nodeListKey == UpdatePolicyBase<Dimension>::wildcard());
  FieldList<Dimension, Scalar> stateFields = state.fields(fieldKey, Scalar());
  const unsigned numFields = stateFields.numFields();

  // Get the mass density, specific thermal energy, and pressure fields
  // from the state.
  const auto massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  const auto energy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  const auto P = state.fields(HydroFieldNames::pressure, 0.0);
  const auto D = state.fields(SolidFieldNames::tensorDamage, SymTensor::zero);
    
  // Walk the individual fields.
  for (auto k = 0u; k != numFields; ++k) {

    // Get the strength model.  This cast is ugly, but is a work-around for now.
    const auto* solidNodeListPtr = dynamic_cast<const SolidNodeList<Dimension>*>(stateFields[k]->nodeListPtr());
    CHECK(solidNodeListPtr != 0);
    const auto& strengthModel = solidNodeListPtr->strengthModel();

    // Now set the shear modulus.
    strengthModel.shearModulus(*stateFields[k], *massDensity[k], *energy[k], *P[k], *D[k]);
  }

//     // Is there a scalar damage field for this NodeList?
//     {
//       const KeyType DKey(nodeListPtr, SolidFieldNames::scalarDamage);
//       if (state.scalarFieldRegistered(DKey)) {
//         const Field<Dimension, Scalar>& D = state.scalarField(DKey);
//         for (int i = 0; i != nodeListPtr->numInternalNodes(); ++i) {
//           CHECK(D(i) >= 0.0 && D(i) <= 1.0);
//           stateField(i) *= 1.0 - D(i);
//         }
//       }
//     }

//     // Is there a tensor damage field for this NodeList?
//     {
//       typedef typename Dimension::SymTensor SymTensor;
//       const KeyType DKey(nodeListPtr, SolidFieldNames::tensorDamage);
//       if (state.symTensorFieldRegistered(DKey)) {
//         const Field<Dimension, SymTensor>& D = state.symTensorField(DKey);
//         for (int i = 0; i != nodeListPtr->numInternalNodes(); ++i) {
//           const Scalar Di = std::max(0.0, std::min(1.0, D(i).eigenValues().maxElement() + 1.0e-5));
//           CHECK(Di >= 0.0 && Di <= 1.0);
//           stateField(i) *= 1.0 - Di;
//         }
//       }
//     }
}

//------------------------------------------------------------------------------
// Equivalence operator.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
ShearModulusPolicy<Dimension>::
operator==(const UpdatePolicyBase<Dimension>& rhs) const {

  // We're only equal if the other guy is also an increment operator.
  const ShearModulusPolicy<Dimension>* rhsPtr = dynamic_cast<const ShearModulusPolicy<Dimension>*>(&rhs);
  if (rhsPtr == 0) {
    return false;
  } else {
    return true;
  }
}

}

