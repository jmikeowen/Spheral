//---------------------------------Spheral++----------------------------------//
// SPHHydroBaseSpherical -- An SPH/ASPH hydrodynamic package for Spheral++,
//                          specialized for 1D Spherical (r) geometry.
//
// Based on the algorithm described in
// Omang, M., Børve, S., & Trulsen, J. (2006). SPH in spherical and cylindrical coordinates.
// Journal of Computational Physics, 213(1), 391412. https://doi.org/10.1016/j.jcp.2005.08.023
//
// Note this version is currently abusing our ordinary 1D geometric types,
// implicitly mapping x->r.
//
// Created by JMO, Tue Dec 22 10:04:21 PST 2020
//----------------------------------------------------------------------------//
#ifndef __Spheral_SPHHydroBaseSpherical_hh__
#define __Spheral_SPHHydroBaseSpherical_hh__

#include <string>

#include "SPHHydroBase.hh"
#include "Geometry/Dimension.hh"

namespace Spheral {

class SPHHydroBaseSpherical: public SPHHydroBase<Dim<1>> {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef Dim<1> Dimension;
  typedef Dimension::Scalar Scalar;
  typedef Dimension::Vector Vector;
  typedef Dimension::Tensor Tensor;
  typedef Dimension::SymTensor SymTensor;

  typedef Physics<Dimension>::ConstBoundaryIterator ConstBoundaryIterator;

  // Constructors.
  SPHHydroBaseSpherical(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
                        DataBase<Dimension>& dataBase,
                        ArtificialViscosity<Dimension>& Q,
                        const TableKernel<Dimension>& W,
                        const TableKernel<Dimension>& WPi,
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
                        const Vector& xmin,
                        const Vector& xmax);

  // Destructor.
  virtual ~SPHHydroBaseSpherical();

  // Tasks we do once on problem startup.
  virtual
  void initializeProblemStartup(DataBase<Dimension>& dataBase) override;

  // // Register the state Hydro expects to use and evolve.
  // virtual 
  // void registerState(DataBase<Dimension>& dataBase,
  //                    State<Dimension>& state) override;

  // // This method is called once at the beginning of a timestep, after all state registration.
  // virtual void preStepInitialize(const DataBase<Dimension>& dataBase, 
  //                                State<Dimension>& state,
  //                                StateDerivatives<Dimension>& derivs) override;

  // Evaluate the derivatives for the principle hydro variables:
  // mass density, velocity, and specific thermal energy.
  virtual
  void evaluateDerivatives(const Scalar time,
                           const Scalar dt,
                           const DataBase<Dimension>& dataBase,
                           const State<Dimension>& state,
                           StateDerivatives<Dimension>& derivatives) const override;

  // // Apply boundary conditions to the physics specific fields.
  // virtual
  // void applyGhostBoundaries(State<Dimension>& state,
  //                           StateDerivatives<Dimension>& derivs) override;

  // // Enforce boundary conditions for the physics specific fields.
  // virtual
  // void enforceBoundaries(State<Dimension>& state,
  //                        StateDerivatives<Dimension>& derivs) override;
               
  //****************************************************************************
  // Methods required for restarting.
  virtual std::string label() const override { return "SPHHydroBaseSpherical" ; }
  //****************************************************************************

private:
  //--------------------------- Private Interface ---------------------------//
  // No default constructor, copying, or assignment.
  SPHHydroBaseSpherical();
  SPHHydroBaseSpherical(const SPHHydroBaseSpherical&);
  SPHHydroBaseSpherical& operator=(const SPHHydroBaseSpherical&);
};

}

#else

// Forward declaration.
namespace Spheral {
  class SPHHydroBaseSpherical;
}

#endif
