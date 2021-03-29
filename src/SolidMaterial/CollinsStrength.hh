//---------------------------------Spheral++----------------------------------//
// CollinsYieldStrength -- Implements a pressure dependent yield strength
// model appropriate for geological materials.
//
// Since this is solely a yield strength model it takes another StrengthModel
// as an argument to compute the shear modulus.  Perhaps at some point we should
// just split up the ideas of what provides shear modulus and yield strength?
// 
//    See Collins, Melosh, Ivanov, 2004 Appendix, MAPS
//
// Created by JMO, Thu Jan 14 16:40:21 PST 2016
//     Based on python implementation by Megan Syal and Cody Raskin
//----------------------------------------------------------------------------//
#ifndef __Spheral_CollinsStrength_hh__
#define __Spheral_CollinsStrength_hh__

#include "StrengthModel.hh"

namespace Spheral {

template<typename Dimension>
class CollinsStrength: public StrengthModel<Dimension> {

public:
  //--------------------------- Public Interface ---------------------------//
  using Scalar = typename Dimension::Scalar;
  using SymTensor = typename Dimension::SymTensor;

  // Constructors, destructor.
  CollinsStrength(const StrengthModel<Dimension>& shearModulusModel,  // Used to compute the shear modulus
                  const double mui,                                   // Coefficient of internal friction (intact material)
                  const double mud,                                   // Coefficient of internal friction (damaged material)
                  const double Y0,                                    // Shear strength at zero pressure
                  const double Ym);                                   // von Mises plastic limit
  // Backwards compatible constructor without the mud term.  To be deprecated...
  CollinsStrength(const StrengthModel<Dimension>& shearModulusModel,  // Used to compute the shear modulus
                  const double mui,                                   // Coefficient of internal friction
                  const double Y0,                                    // Shear strength at zero pressure
                  const double Ym);                                   // von Mises plastic limit
  virtual ~CollinsStrength();


  // Override the required generic interface.
  virtual bool providesSoundSpeed() const override { return mShearModulusModel.providesSoundSpeed(); }

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

  virtual void soundSpeed(Field<Dimension, Scalar>& soundSpeed,
                          const Field<Dimension, Scalar>& density,
                          const Field<Dimension, Scalar>& specificThermalEnergy,
                          const Field<Dimension, Scalar>& pressure,
                          const Field<Dimension, Scalar>& fluidSoundSpeed,
                          const Field<Dimension, SymTensor>& damage) const override;

  // Access the strength parameters.
  const StrengthModel<Dimension>& shearModulusModel() const;
  double mui() const;
  double mud() const;
  double Y0() const;
  double Ym() const;

private:
  //--------------------------- Private Interface ---------------------------//
  const StrengthModel<Dimension>& mShearModulusModel;
  double mmui, mmud, mY0, mYm;

  // No copying or assignment.
  CollinsStrength(const CollinsStrength&);
  CollinsStrength& operator=(const CollinsStrength&);
};

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class CollinsStrength;
}

#endif

