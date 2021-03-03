//---------------------------------Spheral++----------------------------------//
// SteinbergGuinanStrength -- Implements the Steinberg-Guinan strength model.
//   ** Need reference **
//
// Created by JMO, Wed Sep 8 15:18:44 2004
//----------------------------------------------------------------------------//
#ifndef __Spheral_SteinbergGuinanStrength_hh__
#define __Spheral_SteinbergGuinanStrength_hh__

#include "StrengthModel.hh"
#include "PolynomialFit.hh"

namespace Spheral {

class NinthOrderPolynomialFit;
template<typename Dimension> class SolidEquationOfState;

template<typename Dimension>
class SteinbergGuinanStrength: public StrengthModel<Dimension> {

public:
  //--------------------------- Public Interface ---------------------------//
  using Scalar = typename Dimension::Scalar;
  using SymTensor = typename Dimension::SymTensor;

  // Constructors, destructor.
  SteinbergGuinanStrength(const SolidEquationOfState<Dimension>& eos,
                          const double G0,
                          const double Gmax,
                          const double A,
                          const double B,
                          const double Y0,
                          const double Ymax,
                          const double Yp,
                          const double beta,
                          const double gamma0,
                          const double nhard,
                          const NinthOrderPolynomialFit& coldEnergyFit,
                          const NinthOrderPolynomialFit& meltEnergyFit);
  SteinbergGuinanStrength(const SolidEquationOfState<Dimension>& eos,
                          const double G0,
                          const double A,
                          const double B,
                          const double Y0,
                          const double Ymax,
                          const double Yp,
                          const double beta,
                          const double gamma0,
                          const double nhard,
                          const NinthOrderPolynomialFit& coldEnergyFit,
                          const NinthOrderPolynomialFit& meltEnergyFit);
  virtual ~SteinbergGuinanStrength();

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

  // Steinberg-Guinan also can provide melt and cold energies.
  virtual void meltSpecificEnergy(Field<Dimension, Scalar>& meltSpecificEnergy,
                                  const Field<Dimension, Scalar>& density,
                                  const Field<Dimension, Scalar>& specficThermalEnergy) const override;

  virtual void coldSpecificEnergy(Field<Dimension, Scalar>& coldSpecificEnergy,
                                  const Field<Dimension, Scalar>& density,
                                  const Field<Dimension, Scalar>& specficThermalEnergy) const override;

  // Melt attenuation.
  double meltAttenuation(const double rho, const double eps) const;

  // Steinberg-Guinan "temperature".
  void computeTemperature(Field<Dimension, Scalar>& temperature,
                          const Field<Dimension, Scalar>& density,
                          const Field<Dimension, Scalar>& specificThermalEnergy) const;

  // Access the strength parameters.
  double G0() const;
  double Gmax() const;
  double A() const;
  double B() const;
  double Y0() const;
  double Ymax() const;
  double Yp() const;
  double beta() const;
  double gamma0() const;
  double nhard() const;
  const NinthOrderPolynomialFit& coldEnergyFit() const;
  const NinthOrderPolynomialFit& meltEnergyFit() const;

private:
  //--------------------------- Private Interface ---------------------------//
  const SolidEquationOfState<Dimension>* mEOSPtr;
  double mG0;
  double mGmax;
  double mA;
  double mB;
  double mY0;
  double mYmax;
  double mYp;
  double mbeta;
  double mgamma0;
  double mnhard;
  NinthOrderPolynomialFit mColdEnergyFit;
  NinthOrderPolynomialFit mMeltEnergyFit;

  // No copying or assignment.
  SteinbergGuinanStrength(const SteinbergGuinanStrength&);
  SteinbergGuinanStrength& operator=(const SteinbergGuinanStrength&);
};

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class SteinbergGuinanStrength;
}

#endif

