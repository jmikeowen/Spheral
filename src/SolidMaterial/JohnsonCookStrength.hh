//---------------------------------Spheral++----------------------------------//
// JohnsonCookStrength -- Implements the Johnson-Cook strength model.
//   ** Need reference **
//
// Created by JMO, Tue Jul 28 13:39:50 PDT 2015
//----------------------------------------------------------------------------//
#ifndef __Spheral_JohnsonCookStrength_hh__
#define __Spheral_JohnsonCookStrength_hh__

#include "StrengthModel.hh"
#include "PolynomialFit.hh"

namespace Spheral {

template<typename Dimension> class SolidEquationOfState;

template<typename Dimension>
class JohnsonCookStrength: public StrengthModel<Dimension> {

public:
  //--------------------------- Public Interface ---------------------------//
  using Scalar = typename Dimension::Scalar;
  using SymTensor = typename Dimension::SymTensor;

  // Constructors, destructor.
  JohnsonCookStrength(const SolidEquationOfState<Dimension>& eos,
                      const StrengthModel<Dimension>& shearModulusModel,
                      const double A,
                      const double B,
                      const double C,
                      const double C4,
                      const double m,
                      const double nhard,
                      const double epsdot0,
                      const double epsdotmin,
                      const double Tmelt,
                      const double Troom,
                      const double mu0,
                      const bool shearModulusScaling);
  virtual ~JohnsonCookStrength();

  // Override the required generic interface.
  virtual bool providesSoundSpeed() const override { return true; }

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
  double A() const;
  double B() const;
  double C() const;
  double C4() const;
  double m() const;
  double nhard() const;
  double epsdot0() const;
  double epsdotmin() const;
  double Tmelt() const;
  double Troom() const;
  double mu0() const;
  bool shearModulusScaling() const;

private:
  //--------------------------- Private Interface ---------------------------//
  const SolidEquationOfState<Dimension>* mEOSPtr;
  const StrengthModel<Dimension>* mShearModulusModelPtr;
  const double mA, mB, mC, mC4, mm, mnhard, mEpsdot0, mEpsdotmin, mTmelt, mTroom, mmu0;
  bool mShearModulusScaling;

  // No copying or assignment.
  JohnsonCookStrength(const JohnsonCookStrength&);
  JohnsonCookStrength& operator=(const JohnsonCookStrength&);
};

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class JohnsonCookStrength;
}

#endif

