//---------------------------------Spheral++----------------------------------//
// JohnsonCookDamage -- an implementation of a Johnson-Cook damage law.
//
// Created by JMO, Mon Jul  9 08:21:23 PDT 2018
//----------------------------------------------------------------------------//
#ifndef __Spheral_JohnsonCookDamage_hh__
#define __Spheral_JohnsonCookDamage_hh__

#include <vector>

#include "Physics/Physics.hh"
#include "DataOutput/registerWithRestart.hh"

// Forward declarations.
namespace Spheral {
  template<typename Dimension> class State;
  template<typename Dimension> class StateDerivatives;
  template<typename Dimension> class SolidNodeList;
  template<typename Dimension> class DataBase;
  template<typename Dimension, typename DataType> class Field;
  template<typename Dimension, typename DataType> class FieldList;
  template<typename Dimension> class TableKernel;
  class FileIO;
}

namespace Spheral {

template<typename Dimension>
class JohnsonCookDamage: public Physics<Dimension> {

public:
  //--------------------------- Public Interface ---------------------------//
  // Useful typedefs.
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  typedef typename Physics<Dimension>::TimeStepType TimeStepType;
  typedef typename Physics<Dimension>::ConstBoundaryIterator ConstBoundaryIterator;

  // Constructors, destructor.
  JohnsonCookDamage(SolidNodeList<Dimension>& nodeList,
                    const Field<Dimension, Scalar>& D1,
                    const Field<Dimension, Scalar>& D2,
                    const double D3,
                    const double D4,
                    const double D5,
                    const double epsilondot0,
                    const double Tcrit,
                    const double sigmamax,
                    const double efailmin);
  virtual ~JohnsonCookDamage();

  // Attributes.
  const SolidNodeList<Dimension>& nodeList() const;
  const Field<Dimension, Scalar>& failureStrain() const;
  const Field<Dimension, Scalar>& meltSpecificEnergy() const;
  const Field<Dimension, Scalar>& D1() const;
  const Field<Dimension, Scalar>& D2() const;
  double D3() const;
  double D4() const;
  double D5() const;
  double epsilondot0() const;
  double Tcrit() const;
  double sigmamax() const;
  double efailmin() const;

  // Increment the derivatives.
  virtual void evaluateDerivatives(const Scalar time,
                                   const Scalar dt,
                                   const DataBase<Dimension>& dataBase,
                                   const State<Dimension>& state,
                                   StateDerivatives<Dimension>& derivatives) const override;

  // Vote on a time step.
  virtual TimeStepType dt(const DataBase<Dimension>& dataBase, 
                          const State<Dimension>& state,
                          const StateDerivatives<Dimension>& derivs,
                          const Scalar currentTime) const override;

  // Register the state you want carried around (and potentially evolved), as
  // well as the policies for such evolution.
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

  //**************************************************************************
  // Restart methods.
  virtual std::string label() const override { return "JohnsonCookDamage"; }
  virtual void dumpState(FileIO& file, const std::string& pathName) const;
  virtual void restoreState(const FileIO& file, const std::string& pathName);
  //**************************************************************************

private:
  //--------------------------- Private Interface ---------------------------//
  SolidNodeList<Dimension>& mNodeList;
  Field<Dimension, Scalar> mD1, mD2, mFailureStrain, mMeltSpecificEnergy;
  double mD3, mD4, mD5, mepsilondot0, mTcrit, msigmamax, mefailmin;

  // The restart registration.
  RestartRegistrationType mRestart;

  // No default constructor, copying or assignment.
  JohnsonCookDamage();
  JohnsonCookDamage(const JohnsonCookDamage&);
  JohnsonCookDamage& operator=(const JohnsonCookDamage&);
};

}

#include "JohnsonCookDamageInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class JohnsonCookDamage;
}

#endif

