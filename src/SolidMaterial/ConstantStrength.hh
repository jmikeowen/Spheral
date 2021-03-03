//---------------------------------Spheral++----------------------------------//
// ConstantStrength -- An implentation of StrengthModel returning constant
// values for the shear modulus and yield strength.
//
// Created by JMO, Mon Jul 24 10:34:40 PDT 2006
//----------------------------------------------------------------------------//
#ifndef __Spheral_ConstantStrength_hh__
#define __Spheral_ConstantStrength_hh__

#include "StrengthModel.hh"
#include "SolidEquationOfState.hh"
#include "Field/Field.hh"

namespace Spheral {

template<typename Dimension>
class ConstantStrength: public StrengthModel<Dimension> {
public:
  //--------------------------- Public Interface ---------------------------//
  using Scalar = typename Dimension::Scalar;
  using SymTensor = typename Dimension::SymTensor;

  // Constructors, destructor.
  ConstantStrength(const double mu0,
                   const double Y0);
  ConstantStrength(const double mu0,
                   const double Y0,
                   const SolidEquationOfState<Dimension>& eos);
  virtual ~ConstantStrength() {};

  // The generic interface we require all strength models to provide.
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

  // Read only access to the parameters.
  double mu0() const;
  double Y0() const; 

private:
  //--------------------------- Private Interface ---------------------------//
  // The values for shear modulus and yield strength.
  double mShearModulus0;
  double mYieldStrength0;
  const SolidEquationOfState<Dimension>* mEOSptr;

  // No copying or assignment.
  ConstantStrength(const ConstantStrength&);
  ConstantStrength& operator=(const ConstantStrength&);
};

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class ConstantStrength;
}

#endif

