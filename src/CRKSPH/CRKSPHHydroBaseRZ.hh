//---------------------------------Spheral++----------------------------------//
// CRKSPHHydroBaseRZ -- The CRKSPH/ACRKSPH hydrodynamic package for Spheral++.
//
// This is the area-weighted RZ specialization.
//
// Created by JMO, Thu May 12 15:25:24 PDT 2016
//----------------------------------------------------------------------------//
#ifndef __Spheral_CRKSPHHydroBaseRZ_hh__
#define __Spheral_CRKSPHHydroBaseRZ_hh__

#include <string>

#include "CRKSPHHydroBase.hh"
#include "Geometry/Dimension.hh"

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
}

namespace Spheral {

class CRKSPHHydroBaseRZ: public CRKSPHHydroBase<Dim<2> > {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef Dim<2> Dimension;
  typedef Dimension::Scalar Scalar;
  typedef Dimension::Vector Vector;
  typedef Dimension::Tensor Tensor;
  typedef Dimension::ThirdRankTensor ThirdRankTensor;
  typedef Dimension::FourthRankTensor FourthRankTensor;
  typedef Dimension::FifthRankTensor FifthRankTensor;
  typedef Dimension::SymTensor SymTensor;
  typedef Dimension::FacetedVolume FacetedVolume;

  typedef Physics<Dimension>::ConstBoundaryIterator ConstBoundaryIterator;

  // Constructors.
  CRKSPHHydroBaseRZ(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
                    ArtificialViscosity<Dimension>& Q,
                    const TableKernel<Dimension>& W,
                    const TableKernel<Dimension>& WPi,
                    const double filter,
                    const double cfl,
                    const bool useVelocityMagnitudeForDt,
                    const bool compatibleEnergyEvolution,
                    const bool evolveTotalEnergy,
                    const bool XSPH,
                    const MassDensityType densityUpdate,
                    const HEvolutionType HUpdate,
                    const CRKOrder correctionOrder,
                    const CRKVolumeType volumeType,
                    const double epsTensile,
                    const double nTensile,
                    const bool limitMultimaterialTopology);

  // Destructor.
  virtual ~CRKSPHHydroBaseRZ();

  // Tasks we do once on problem startup.
  virtual
  void initializeProblemStartup(DataBase<Dimension>& dataBase) override;

  // Register the state Hydro expects to use and evolve.
  virtual 
  void registerState(DataBase<Dimension>& dataBase,
                     State<Dimension>& state) override;

  // This method is called once at the beginning of a timestep, after all state registration.
  virtual void preStepInitialize(const DataBase<Dimension>& dataBase, 
                                 State<Dimension>& state,
                                 StateDerivatives<Dimension>& derivs) override;

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

  //****************************************************************************
  // Methods required for restarting.
  virtual std::string label() const { return "CRKSPHHydroBaseRZ"; }
  //****************************************************************************

private:
  //--------------------------- Private Interface ---------------------------//
  // No default constructor, copying, or assignment.
  CRKSPHHydroBaseRZ();
  CRKSPHHydroBaseRZ(const CRKSPHHydroBaseRZ&);
  CRKSPHHydroBaseRZ& operator=(const CRKSPHHydroBaseRZ&);
};

}

#else

// Forward declaration.
namespace Spheral {
  class CRKSPHHydroBaseRZ;
}

#endif
