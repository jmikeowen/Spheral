//---------------------------------Spheral++----------------------------------//
// SolidNodeList -- A form of the SPH NodeList appropriate for use with 
// solid materials.
//
// Created by JMO, Tue Sep 7 22:44:37 2004
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "Field/Field.hh"
#include "Field/FieldList.hh"
#include "Material/EquationOfState.hh"
#include "Kernel/TableKernel.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Strength/SolidFieldNames.hh"
#include "DataBase/IncrementState.hh"
#include "SolidMaterial/StrengthModel.hh"
#include "Utilities/DBC.hh"

#include "SolidNodeList.hh"

using std::vector;
using std::list;
using std::string;
using std::cerr;
using std::endl;

namespace Spheral {

//------------------------------------------------------------------------------
// Construct with the given EOS object, along with optional numInternal nodes,
// numGhost nodes, and name.
//------------------------------------------------------------------------------
template<typename Dimension>
SolidNodeList<Dimension>::
SolidNodeList(string name,
              EquationOfState<Dimension>& eos,
              StrengthModel<Dimension>& strength,
              const int numInternal,
              const int numGhost,
              const Scalar hmin,
              const Scalar hmax,
              const Scalar hminratio,
              const Scalar nPerh,
              const int maxNumNeighbors,
              const Scalar rhoMin,
              const Scalar rhoMax):
  FluidNodeList<Dimension>(name, 
                           eos,
                           numInternal, 
                           numGhost,
                           hmin,
                           hmax,
                           hminratio,
                           nPerh,
                           maxNumNeighbors,
                           rhoMin,
                           rhoMax),
  mDeviatoricStress(SolidFieldNames::deviatoricStress, *this),
  mPlasticStrain(SolidFieldNames::plasticStrain, *this),
  mPlasticStrainRate(SolidFieldNames::plasticStrainRate, *this),
  mDamage(SolidFieldNames::tensorDamage, *this),
  mFragmentIDs(SolidFieldNames::fragmentIDs, *this),
  mParticleTypes(SolidFieldNames::particleTypes, *this),
  mStrength(strength) {
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
SolidNodeList<Dimension>::
~SolidNodeList() {
}

//------------------------------------------------------------------------------
// Calculate and return the sound speed field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidNodeList<Dimension>::
soundSpeed(Field<Dimension, typename Dimension::Scalar>& field) const {

  // Get the straight EOS look up from the base class.
  FluidNodeList<Dimension>::soundSpeed(field);

  // Augment the sound speed with the strength model.
  const auto& rho = this->massDensity();
  const auto& u = this->specificThermalEnergy();
  const auto& D = this->damage();
  Field<Dimension, Scalar> P(HydroFieldNames::pressure, *this);
  this->pressure(P);
  mStrength.soundSpeed(field, rho, u, P, field, D);
}

//------------------------------------------------------------------------------
// Calculate and return the bulk modulus.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidNodeList<Dimension>::
bulkModulus(Field<Dimension, typename Dimension::Scalar>& field) const {
  const auto& rho = this->massDensity();
  const auto& u = this->specificThermalEnergy();
  this->equationOfState().setBulkModulus(field, rho, u);
}

//------------------------------------------------------------------------------
// Calculate and return the shear modulus.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidNodeList<Dimension>::
shearModulus(Field<Dimension, typename Dimension::Scalar>& field) const {
  const auto& rho = this->massDensity();
  const auto& u = this->specificThermalEnergy();
  const auto& D = this->damage();
  Field<Dimension, Scalar> P(HydroFieldNames::pressure, *this);
  this->pressure(P);
  mStrength.shearModulus(field, rho, u, P, D);
}

//------------------------------------------------------------------------------
// Calculate and return the yield strength.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidNodeList<Dimension>::
yieldStrength(Field<Dimension, typename Dimension::Scalar>& field) const {
  const auto& rho = this->massDensity();
  const auto& u = this->specificThermalEnergy();
  const auto& D = this->damage();
  Field<Dimension, Scalar> P(HydroFieldNames::pressure, *this);
  this->pressure(P);
  mStrength.yieldStrength(field, rho, u, P, mPlasticStrain, mPlasticStrainRate, D);
}

//------------------------------------------------------------------------------
// Dump the current state of the NodeList to the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidNodeList<Dimension>::
dumpState(FileIO& file, const string& pathName) const {

  // Dump the ancestor class.
  FluidNodeList<Dimension>::dumpState(file, pathName);

  file.write(mDeviatoricStress, pathName + "/" + mDeviatoricStress.name());
  file.write(mPlasticStrain, pathName + "/" + mPlasticStrain.name());
  file.write(mPlasticStrainRate, pathName + "/" + mPlasticStrainRate.name());
  file.write(mDamage, pathName + "/" + mDamage.name());
  file.write(mFragmentIDs, pathName + "/" + mFragmentIDs.name());
  file.write(mParticleTypes, pathName + "/" + mParticleTypes.name());
}

//------------------------------------------------------------------------------
// Restore the state of the NodeList from the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidNodeList<Dimension>::
restoreState(const FileIO& file, const string& pathName) {

  // Restore the ancestor class.
  FluidNodeList<Dimension>::restoreState(file, pathName);

  file.read(mDeviatoricStress, pathName + "/" + mDeviatoricStress.name());
  file.read(mPlasticStrain, pathName + "/" + mPlasticStrain.name());
  file.read(mPlasticStrainRate, pathName + "/" + mPlasticStrainRate.name());
  file.read(mDamage, pathName + "/" + mDamage.name());
  file.read(mFragmentIDs, pathName + "/" + mFragmentIDs.name());
  file.read(mParticleTypes, pathName + "/" + mParticleTypes.name());
}

}
