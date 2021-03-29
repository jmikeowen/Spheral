//---------------------------------Spheral++----------------------------------//
// StrengthModel -- The interface base class for strength models.
//
// Created by JMO, Wed Sep 8 15:18:44 2004
//----------------------------------------------------------------------------//
#ifndef __Spheral_StrengthModel_hh__
#define __Spheral_StrengthModel_hh__

#include "Utilities/SpheralFunctions.hh"
#include "Utilities/DBC.hh"

// Forward declarations.
namespace Spheral {

template<typename Dimension, typename DataType> class Field;

template<typename Dimension>
class StrengthModel {
public:
  //--------------------------- Public Interface ---------------------------//
  using Scalar = typename Dimension::Scalar;
  using SymTensor = typename Dimension::SymTensor;

  // Constructors, destructor.
  StrengthModel();
  virtual ~StrengthModel();

  //............................................................................
  // The generic interface we require all strength models to provide.
  virtual void shearModulus(Field<Dimension, Scalar>& shearModulus,
                            const Field<Dimension, Scalar>& density,
                            const Field<Dimension, Scalar>& specificThermalEnergy,
                            const Field<Dimension, Scalar>& pressure,
                            const Field<Dimension, SymTensor>& damage) const = 0;

  virtual void yieldStrength(Field<Dimension, Scalar>& yieldStrength,
                             const Field<Dimension, Scalar>& density,
                             const Field<Dimension, Scalar>& specificThermalEnergy,
                             const Field<Dimension, Scalar>& pressure,
                             const Field<Dimension, Scalar>& plasticStrain,
                             const Field<Dimension, Scalar>& plasticStrainRate,
                             const Field<Dimension, SymTensor>& damage) const = 0;

  //............................................................................
  // Some strength models optionally provide the following methods.
  virtual bool providesSoundSpeed() const { return false; }
  virtual bool providesBulkModulus() const { return false; }
  virtual void soundSpeed(Field<Dimension, Scalar>& soundSpeed,
                          const Field<Dimension, Scalar>& density,
                          const Field<Dimension, Scalar>& specificThermalEnergy,
                          const Field<Dimension, Scalar>& pressure,
                          const Field<Dimension, Scalar>& fluidSoundSpeed,
                          const Field<Dimension, SymTensor>& damage) const;

  virtual void bulkModulus(Field<Dimension, Scalar>& bulkModulus,
                           const Field<Dimension, Scalar>& massDensity,
                           const Field<Dimension, Scalar>& specificThermalEnergy) const;

  virtual void meltSpecificEnergy(Field<Dimension, Scalar>& meltSpecificEnergy,
                                  const Field<Dimension, Scalar>& density,
                                  const Field<Dimension, Scalar>& specficThermalEnergy) const;

  virtual void coldSpecificEnergy(Field<Dimension, Scalar>& coldSpecificEnergy,
                                  const Field<Dimension, Scalar>& density,
                                  const Field<Dimension, Scalar>& specficThermalEnergy) const;

private:
  //--------------------------- Private Interface ---------------------------//
  // No copying or assignment.
  StrengthModel(const StrengthModel&);
  StrengthModel& operator=(const StrengthModel&);
};

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class StrengthModel;
}

#endif

