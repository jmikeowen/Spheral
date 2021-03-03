//---------------------------------Spheral++----------------------------------//
// TensorDamageModel -- Base class for the tensor damage physics models.
// This class does not know how to seed the flaw distribution -- that is 
// required of descendant classes.
//
// References:
//   Benz, W. & Asphaug, E., 1995 "Computer Physics Comm.", 87, 253-265.
//   Benz, W. & Asphaug, E., 1994 "Icarus", 107, 98-116.
//   Randles, P.W. & Libersky, L.D., 1996, "Comput. Methods Appl. Engrg, 
//     139, 375-408
//
// Created by JMO, Thu Sep 29 15:42:05 PDT 2005
//----------------------------------------------------------------------------//
#ifndef __Spheral_TensorDamageModel_hh__
#define __Spheral_TensorDamageModel_hh__

#include "DamageModel.hh"

#include <vector>

// Forward declarations.
namespace Spheral {
  template<typename Dimension> class State;
  template<typename Dimension> class StateDerivatives;
  template<typename Dimension> class SolidNodeList;
  template<typename Dimension> class DataBase;
  template<typename Dimension, typename DataType> class Field;
  template<typename Dimension, typename DataType> class FieldList;
  class FileIO;
}

namespace Spheral {

// Enum for selecting the method of defining the tensor strain.
enum class TensorStrainAlgorithm {
  BenzAsphaugStrain = 0,
  StrainHistory = 1,
  MeloshRyanAsphaugStrain = 2,
  PlasticStrain = 3,
  PseudoPlasticStrain = 4,
};

// Enum for selecting the method of defining the effective tensor damage.
enum class EffectiveDamageAlgorithm {
  CopyDamage = 0,
  MaxDamage = 1,
  MinMaxDamage = 2,
  SampledDamage = 3,
};

template<typename Dimension>
class TensorDamageModel: 
    public DamageModel<Dimension> {

public:
  //--------------------------- Public Interface ---------------------------//
  // Useful typedefs.
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  typedef typename Physics<Dimension>::TimeStepType TimeStepType;
  typedef typename Physics<Dimension>::ConstBoundaryIterator ConstBoundaryIterator;
  typedef Field<Dimension, std::vector<double> > FlawStorageType;

  // Constructors, destructor.
  TensorDamageModel(SolidNodeList<Dimension>& nodeList,
                    const TensorStrainAlgorithm strainAlgorithm,
                    const DamageCouplingAlgorithm damageCouplingAlgorithm,
                    const TableKernel<Dimension>& W,
                    const double crackGrowthMultiplier,
                    const double criticalDamageThreshold,
                    const bool damageInCompression,
                    const FlawStorageType& flaws);
  virtual ~TensorDamageModel();

  //...........................................................................
  // Provide the required physics package interface.
  // Compute the derivatives.
  virtual 
  void evaluateDerivatives(const Scalar time,
                           const Scalar dt,
                           const DataBase<Dimension>& dataBase,
                           const State<Dimension>& state,
                           StateDerivatives<Dimension>& derivatives) const override;

  // Vote on a time step.
  virtual TimeStepType dt(const DataBase<Dimension>& dataBase, 
                          const State<Dimension>& state,
                          const StateDerivatives<Dimension>& derivs,
                          const Scalar currentTime) const override;

  // Register our state.
  virtual void registerState(DataBase<Dimension>& dataBase,
                             State<Dimension>& state) override;

  // Register the derivatives/change fields for updating state.
  virtual void registerDerivatives(DataBase<Dimension>& dataBase,
                                   StateDerivatives<Dimension>& derivs) override;

  // Apply boundary conditions to the physics specific fields.
  virtual void applyGhostBoundaries(State<Dimension>& state,
                                    StateDerivatives<Dimension>& derivs) override;

  // Enforce boundary conditions for the physics specific fields.
  virtual void enforceBoundaries(State<Dimension>& state,
                                 StateDerivatives<Dimension>& derivs) override;
  //...........................................................................

  // Provide access to the state fields we maintain.
  const Field<Dimension, SymTensor>& strain() const;
  const Field<Dimension, SymTensor>& effectiveStrain() const;
  const Field<Dimension, Scalar>& DdamageDt() const;

  // The algorithms to update the strain.
  TensorStrainAlgorithm strainAlgorithm() const;

  // Flag to determine if damage in compression is allowed.
  bool damageInCompression() const;
  void damageInCompression(bool x);

  // The critical damage threshold for not setting the time step.
  double criticalDamageThreshold() const; 
  void criticalDamageThreshold(double x);

  //**************************************************************************
  // Restart methods.
  virtual std::string label() const { return "TensorDamageModel"; }
  virtual void dumpState(FileIO& file, const std::string& pathName) const;
  virtual void restoreState(const FileIO& file, const std::string& pathName);
  //**************************************************************************

protected:
  //--------------------------- Protected Interface ---------------------------//
  Field<Dimension, SymTensor> mStrain;
  Field<Dimension, SymTensor> mEffectiveStrain;
  Field<Dimension, Scalar> mDdamageDt;

private:
  //--------------------------- Private Interface ---------------------------//
  TensorStrainAlgorithm mStrainAlgorithm;
  double mCriticalDamageThreshold;
  bool mDamageInCompression;

  // No default constructor, copying or assignment.
  TensorDamageModel();
  TensorDamageModel(const TensorDamageModel&);
  TensorDamageModel& operator=(const TensorDamageModel&);
};

}

#include "TensorDamageModelInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class TensorDamageModel;
}

#endif

