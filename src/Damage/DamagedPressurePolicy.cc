//---------------------------------Spheral++----------------------------------//
// DamagedPressurePolicy -- Override the default pressure policy in the 
// presence of damage.
//----------------------------------------------------------------------------//
#include "DamagedPressurePolicy.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Strength/SolidFieldNames.hh"
#include "NodeList/SolidNodeList.hh"
#include "DataBase/UpdatePolicyBase.hh"
#include "DataBase/IncrementState.hh"
#include "DataBase/ReplaceState.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "Field/FieldList.hh"
#include "NodeList/FluidNodeList.hh"
#include "Material/EquationOfState.hh"
#include "Utilities/DBC.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
DamagedPressurePolicy<Dimension>::
DamagedPressurePolicy():
  PressurePolicy<Dimension>() {
  this->addDependency(SolidFieldNames::tensorDamage);
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
DamagedPressurePolicy<Dimension>::
~DamagedPressurePolicy() {
}

//------------------------------------------------------------------------------
// Update the field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DamagedPressurePolicy<Dimension>::
update(const KeyType& key,
       State<Dimension>& state,
       StateDerivatives<Dimension>& derivs,
       const double multiplier,
       const double t,
       const double dt) {
  KeyType fieldKey, nodeListKey, Dkey;
  StateBase<Dimension>::splitFieldKey(key, fieldKey, nodeListKey);
  REQUIRE(fieldKey == HydroFieldNames::pressure and 
          nodeListKey == UpdatePolicyBase<Dimension>::wildcard());
  StateBase<Dimension>::buildFieldKey(SolidFieldNames::tensorDamage, nodeListKey);
  auto pressure = state.fields(fieldKey, 0.0);
  auto D = state.fields(SolidFieldNames::tensorDamage, SymTensor::zero);
  const auto numFields = pressure.numFields();
  REQUIRE(D.numFields() == numFields);

  // Have the base class set the initial pressure.
  PressurePolicy<Dimension>::update(key, state, derivs, multiplier, t, dt);

  // Scale by the damage.
  for (auto il = 0u; il < numFields; ++il) {
    const auto ni = pressure[il]->numInternalElements();
#pragma omp parallel for
    for (auto i = 0u; i < ni; ++i) {
      if (pressure(il,i) < 0.0) {
        const Scalar fDi = std::max(0.0, std::min(1.0, 1.0 - D(il,i).Trace()/Dimension::nDim));
        CHECK(fDi >= 0.0 and fDi <= 1.0);
        pressure(il,i) *= fDi;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Equivalence operator.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
DamagedPressurePolicy<Dimension>::
operator==(const UpdatePolicyBase<Dimension>& rhs) const {

  // We're only equal if the other guy is also an increment operator.
  const DamagedPressurePolicy<Dimension>* rhsPtr = dynamic_cast<const DamagedPressurePolicy<Dimension>*>(&rhs);
  if (rhsPtr == 0) {
    return false;
  } else {
    return true;
  }
}

}
