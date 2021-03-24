//---------------------------------Spheral++----------------------------------//
// SolidSPHHydroBaseR -- The spherical (R) SPH/ASPH solid material
//                       hydrodynamic package for Spheral++.
//
// This spherical version implements a solid version of the algorithm from
// 
// Omang, M., Børve, S., & Trulsen, J. (2006). SPH in spherical and cylindrical
// coordinates. Journal of Computational Physics, 213(1), 391–412.
// https://doi.org/10.1016/j.jcp.2005.08.023
//
// Note this version abuses our ordinary 1D geometric types, implicitly
// mapping x->r.
//
// Created by JMO, Wed Mar 10 13:19:40 PST 2021
//----------------------------------------------------------------------------//
#ifndef __Spheral_SolidSPHHydroBaseR_hh__
#define __Spheral_SolidSPHHydroBaseR_hh__

#include <float.h>
#include <string>

#include "SolidSPHHydroBase.hh"
#include "Geometry/Dimension.hh"
#include "Kernel/SphericalTableKernel.hh"

namespace Spheral {

template<typename Dimension> class State;
template<typename Dimension> class StateDerivatives;
template<typename Dimension> class SmoothingScaleBase;
template<typename Dimension> class ArtificialViscosity;
template<typename Dimension> class TableKernel;
template<typename Dimension> class DataBase;
template<typename Dimension, typename DataType> class Field;
template<typename Dimension, typename DataType> class FieldList;
class FileIO;

class SolidSPHHydroBaseR: public SolidSPHHydroBase<Dim<1> > {

public:
  //--------------------------- Public Interface ---------------------------//
  using Dimension = Dim<1>;
  using Scalar = Dimension::Scalar;
  using Vector = Dimension::Vector;
  using Tensor = Dimension::Tensor;
  using SymTensor = Dimension::SymTensor;
  using ConstBoundaryIterator = Physics<Dimension>::ConstBoundaryIterator;

  // Constructors.
  SolidSPHHydroBaseR(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
                     DataBase<Dimension>& dataBase,
                     ArtificialViscosity<Dimension>& Q,
                     const SphericalTableKernel& W,
                     const SphericalTableKernel& WPi,
                     const SphericalTableKernel& WGrad,
                     const double filter,
                     const double cfl,
                     const bool useVelocityMagnitudeForDt,
                     const bool compatibleEnergyEvolution,
                     const bool evolveTotalEnergy,
                     const bool gradhCorrection,
                     const bool XSPH,
                     const bool correctVelocityGradient,
                     const bool sumMassDensityOverAllNodeLists,
                     const MassDensityType densityUpdate,
                     const HEvolutionType HUpdate,
                     const double epsTensile,
                     const double nTensile,
                     const bool damageRelieveRubble,
                     const bool negativePressureInDamage,
                     const bool strengthInDamage,
                     const Vector& xmin,
                     const Vector& xmax);

  // Destructor.
  virtual ~SolidSPHHydroBaseR();

  // Tasks we do once on problem startup.
  virtual
  void initializeProblemStartup(DataBase<Dimension>& dataBase) override;

  // Register the state Hydro expects to use and evolve.
  virtual 
  void registerState(DataBase<Dimension>& dataBase,
                     State<Dimension>& state) override;

  // Evaluate the derivatives for the principle hydro variables:
  // mass density, velocity, and specific thermal energy.
  virtual
  void evaluateDerivatives(const Scalar time,
                           const Scalar dt,
                           const DataBase<Dimension>& dataBase,
                           const State<Dimension>& state,
                           StateDerivatives<Dimension>& derivatives) const override;

  // Apply boundary conditions to the physics specific fields.
  virtual
  void applyGhostBoundaries(State<Dimension>& state,
                            StateDerivatives<Dimension>& derivs) override;

  // Enforce boundary conditions for the physics specific fields.
  virtual
  void enforceBoundaries(State<Dimension>& state,
                         StateDerivatives<Dimension>& derivs) override;

  // Access the spherical kernels.
  const SphericalTableKernel& sphericalKernel() const;
  const SphericalTableKernel& sphericalPiKernel() const;
  const SphericalTableKernel& sphericalGradKernel() const;

  //****************************************************************************
  // Methods required for restarting.
  virtual std::string label() const override { return "SolidSPHHydroBaseR"; }

private:
  //--------------------------- Private Interface ---------------------------//
  // The special spherical kernel(s).
  const SphericalTableKernel *mKernelPtr, *mPiKernelPtr, *mGKernelPtr;

  // No default constructor, copying, or assignment.
  SolidSPHHydroBaseR();
  SolidSPHHydroBaseR(const SolidSPHHydroBaseR&);
  SolidSPHHydroBaseR& operator=(const SolidSPHHydroBaseR&);
};

}

#else

// Forward declaration.
namespace Spheral {
  class SolidSPHHydroBaseR;
}

#endif
