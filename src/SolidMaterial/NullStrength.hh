//---------------------------------Spheral++----------------------------------//
// NullStrength -- An implentation of StrengthModel which mimics having
// no strength.
//
// Created by JMO, Thu Sep 16 23:48:47 PDT 2004
//----------------------------------------------------------------------------//
#ifndef __Spheral_NullStrength_hh__
#define __Spheral_NullStrength_hh__

#include "StrengthModel.hh"
#include "Field/Field.hh"

namespace Spheral {

template<typename Dimension>
class NullStrength: public StrengthModel<Dimension> {
public:
  //--------------------------- Public Interface ---------------------------//
  using Scalar = typename Dimension::Scalar;
  using SymTensor = typename Dimension::SymTensor;

  // Constructors, destructor.
  NullStrength() {};
  virtual ~NullStrength() {};

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

private:
  //--------------------------- Private Interface ---------------------------//

  // No copying or assignment.
  NullStrength(const NullStrength&);
  NullStrength& operator=(const NullStrength&);
};

//------------------------------------------------------------------------------
// Compute the shear modulus.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
void
NullStrength<Dimension>::
shearModulus(Field<Dimension, Scalar>& shearModulus,
             const Field<Dimension, Scalar>& /*density*/,
             const Field<Dimension, Scalar>& /*specificThermalEnergy*/,
             const Field<Dimension, Scalar>& /*pressure*/,
             const Field<Dimension, SymTensor>& /*damage*/) const {
  shearModulus = 0.0;
}

//------------------------------------------------------------------------------
// Compute the yield strength.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
void
NullStrength<Dimension>::
yieldStrength(Field<Dimension, Scalar>& yieldStrength,
              const Field<Dimension, Scalar>& /*density*/,
              const Field<Dimension, Scalar>& /*specificThermalEnergy*/,
              const Field<Dimension, Scalar>& /*pressure*/,
              const Field<Dimension, Scalar>& /*plasticStrain*/,
              const Field<Dimension, Scalar>& /*plasticStrainRate*/,
              const Field<Dimension, SymTensor>& /*damage*/) const {
  yieldStrength = 0.0;
}

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class NullStrength;
}

#endif

