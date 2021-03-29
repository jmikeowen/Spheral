//---------------------------------Spheral++----------------------------------//
// PorousStrengthModel
//
// An implementation of strain-alpha porosity model described in
// Wunnemann, Collins, & Melosh, Icarus, 180, 514-527 (2006)
// "A strain-based porosity model for use in hydrocode simulations of impacts
//  and implications for transient crater growth in porous targets"
//
// This model assumes you will provide a solid EOS which will be modified.
// The underlying actualy solid EOS should provide the reference density, which
// will be treated here as the compacted true solid reference density.
//
// Note this model introduces a new state variable, the distention (alpha), which
// the pressure now depends on.  This implies our usual definition of P(rho, eps)
// now becomes P(rho, eps, alpha).  Our EOS interface does not recognize this
// this parameter, so we store alpha locally and only allow Field updates of the
// pressure (forbidding the single value P lookup the EOS usually allows).
//
// Created by JMO, Thu Sep 13 15:50:28 PDT 2012
//----------------------------------------------------------------------------//
#ifndef __Spheral_PorousStrengthModel_hh__
#define __Spheral_PorousStrengthModel_hh__

#include "StrengthModel.hh"
#include "Utilities/SpheralFunctions.hh"
#include "Utilities/DBC.hh"

namespace Spheral {

template<typename Dimension>
class PorousStrengthModel: public StrengthModel<Dimension> {
public:
  //--------------------------- Public Interface ---------------------------//
  using Scalar = typename Dimension::Scalar;
  using SymTensor = typename Dimension::SymTensor;

  // Constructors, destructor.
  PorousStrengthModel(const StrengthModel<Dimension>& solidStrength);
  virtual ~PorousStrengthModel();

  // We require the Field only interface!
  virtual void shearModulus(Field<Dimension, Scalar>& shearModulus,
                            const Field<Dimension, Scalar>& density,
                            const Field<Dimension, Scalar>& specificThermalEnergy,
                            const Field<Dimension, Scalar>& pressure,
                            const Field<Dimension, SymTensor>& damage) const override;

  virtual void yieldStrength(Field<Dimension, Scalar>& yieldStrength,
                             const Field<Dimension, Scalar>& density,
                             const Field<Dimension, Scalar>& specificThermalEnergy,
                             const Field<Dimension, Scalar>& pressure,
                             const Field<Dimension, Scalar>& plasticStrain,
                             const Field<Dimension, Scalar>& plasticStrainRate,
                             const Field<Dimension, SymTensor>& damage) const override;

  // The optional methods the underlying strength model might provide.
  virtual bool providesSoundSpeed() const override { return mSolidStrength.providesSoundSpeed(); }
  virtual bool providesBulkModulus() const override { return mSolidStrength.providesBulkModulus(); }
  virtual void soundSpeed(Field<Dimension, Scalar>& soundSpeed,
                          const Field<Dimension, Scalar>& density,
                          const Field<Dimension, Scalar>& specificThermalEnergy,
                          const Field<Dimension, Scalar>& pressure,
                          const Field<Dimension, Scalar>& fluidSoundSpeed,
                          const Field<Dimension, SymTensor>& damage) const override;

  virtual void bulkModulus(Field<Dimension, Scalar>& bulkModulus,
                           const Field<Dimension, Scalar>& massDensity,
                           const Field<Dimension, Scalar>& specificThermalEnergy) const override;

  virtual void meltSpecificEnergy(Field<Dimension, Scalar>& meltSpecificEnergy,
                                  const Field<Dimension, Scalar>& density,
                                  const Field<Dimension, Scalar>& specficThermalEnergy) const override;

  virtual void coldSpecificEnergy(Field<Dimension, Scalar>& coldSpecificEnergy,
                                  const Field<Dimension, Scalar>& density,
                                  const Field<Dimension, Scalar>& specficThermalEnergy) const override;

  // Access the material parameters.
  const StrengthModel<Dimension>& solidStrength() const;
  const Field<Dimension, Scalar>& alpha() const;
  void alpha(const Field<Dimension, Scalar>& x);

private:
  //--------------------------- Private Interface ---------------------------//
  const StrengthModel<Dimension>& mSolidStrength;
  const Field<Dimension, Scalar>* mAlphaPtr;

  // No copying or assignment.
  PorousStrengthModel(const PorousStrengthModel&);
  PorousStrengthModel& operator=(const PorousStrengthModel&);
};

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class PorousStrengthModel;
}

#endif

