//---------------------------------Spheral++----------------------------------//
// SPHHydroBase -- The SPH/ASPH hydrodynamic package for Spheral++.
//
// Created by JMO, Mon Jul 19 22:11:09 PDT 2010
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "computeSPHSumMassDensity.hh"
#include "correctSPHSumMassDensity.hh"
#include "computeSumVoronoiCellMassDensity.hh"
#include "computeSPHOmegaGradhCorrection.hh"
#include "NodeList/SmoothingScaleBase.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Physics/GenericHydro.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/IncrementFieldList.hh"
#include "DataBase/ReplaceFieldList.hh"
#include "DataBase/IncrementBoundedFieldList.hh"
#include "DataBase/ReplaceBoundedFieldList.hh"
#include "DataBase/IncrementBoundedState.hh"
#include "DataBase/ReplaceBoundedState.hh"
#include "DataBase/CompositeFieldListPolicy.hh"
#include "Hydro/VolumePolicy.hh"
#include "Hydro/VoronoiMassDensityPolicy.hh"
#include "Hydro/SumVoronoiMassDensityPolicy.hh"
#include "Hydro/SpecificThermalEnergyPolicy.hh"
#include "Hydro/SpecificFromTotalThermalEnergyPolicy.hh"
#include "Hydro/PositionPolicy.hh"
#include "Hydro/PressurePolicy.hh"
#include "Hydro/SoundSpeedPolicy.hh"
#include "Hydro/EntropyPolicy.hh"
#include "Mesh/MeshPolicy.hh"
#include "Mesh/generateMesh.hh"
#include "ArtificialViscosity/ArtificialViscosity.hh"
#include "DataBase/DataBase.hh"
#include "Field/FieldList.hh"
#include "Field/NodeIterators.hh"
#include "Boundary/Boundary.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Utilities/timingUtilities.hh"
#include "Utilities/safeInv.hh"
#include "Utilities/globalBoundingVolumes.hh"
#include "Utilities/Timer.hh"
#include "Mesh/Mesh.hh"
#include "CRKSPH/volumeSpacing.hh"

#include "SPHHydroBase.hh"

#ifdef _OPENMP
#include "omp.h"
#endif

#include <limits.h>
#include <float.h>
#include <algorithm>
#include <fstream>
#include <map>
#include <vector>
using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::abs;

// Declare timers
extern Timer TIME_SPH;
extern Timer TIME_SPHinitializeStartup;
extern Timer TIME_SPHregister;
extern Timer TIME_SPHregisterDerivs;
extern Timer TIME_SPHpreStepInitialize;
extern Timer TIME_SPHinitialize;
extern Timer TIME_SPHfinalizeDerivs;
extern Timer TIME_SPHghostBounds;
extern Timer TIME_SPHupdateVol;
extern Timer TIME_SPHenforceBounds;
extern Timer TIME_SPHevalDerivs;
extern Timer TIME_SPHevalDerivs_initial;
extern Timer TIME_SPHevalDerivs_pairs;
extern Timer TIME_SPHevalDerivs_final;

namespace Spheral {

//------------------------------------------------------------------------------
// Construct with the given artificial viscosity and kernels.
//------------------------------------------------------------------------------
template<typename Dimension>
SPHHydroBase<Dimension>::
SPHHydroBase(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
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
             const Vector& xmax):
  GenericHydro<Dimension>(Q, cfl, useVelocityMagnitudeForDt),
  mKernel(W),
  mPiKernel(WPi),
  mSmoothingScaleMethod(smoothingScaleMethod),
  mDensityUpdate(densityUpdate),
  mHEvolution(HUpdate),
  mCompatibleEnergyEvolution(compatibleEnergyEvolution),
  mEvolveTotalEnergy(evolveTotalEnergy),
  mGradhCorrection(gradhCorrection),
  mXSPH(XSPH),
  mCorrectVelocityGradient(correctVelocityGradient),
  mSumMassDensityOverAllNodeLists(sumMassDensityOverAllNodeLists),
  mfilter(filter),
  mEpsTensile(epsTensile),
  mnTensile(nTensile),
  mxmin(xmin),
  mxmax(xmax),
  mTimeStepMask(FieldStorageType::CopyFields),
  mPressure(FieldStorageType::CopyFields),
  mSoundSpeed(FieldStorageType::CopyFields),
  mOmegaGradh(FieldStorageType::CopyFields),
  mSpecificThermalEnergy0(FieldStorageType::CopyFields),
  mEntropy(FieldStorageType::CopyFields),
  mHideal(FieldStorageType::CopyFields),
  mMaxViscousPressure(FieldStorageType::CopyFields),
  mEffViscousPressure(FieldStorageType::CopyFields),
  mMassDensityCorrection(FieldStorageType::CopyFields),
  mViscousWork(FieldStorageType::CopyFields),
  mMassDensitySum(FieldStorageType::CopyFields),
  mNormalization(FieldStorageType::CopyFields),
  mWeightedNeighborSum(FieldStorageType::CopyFields),
  mMassSecondMoment(FieldStorageType::CopyFields),
  mXSPHWeightSum(FieldStorageType::CopyFields),
  mXSPHDeltaV(FieldStorageType::CopyFields),
  mDxDt(FieldStorageType::CopyFields),
  mDvDt(FieldStorageType::CopyFields),
  mDmassDensityDt(FieldStorageType::CopyFields),
  mDspecificThermalEnergyDt(FieldStorageType::CopyFields),
  mDHDt(FieldStorageType::CopyFields),
  mDvDx(FieldStorageType::CopyFields),
  mInternalDvDx(FieldStorageType::CopyFields),
  mM(FieldStorageType::CopyFields),
  mLocalM(FieldStorageType::CopyFields),
  mVolume(FieldStorageType::CopyFields),
  mPairAccelerations(),
  mRestart(registerWithRestart(*this)) {

  // Create storage for our internal state.
  mTimeStepMask = dataBase.newFluidFieldList(int(0), HydroFieldNames::timeStepMask);
  mPressure = dataBase.newFluidFieldList(0.0, HydroFieldNames::pressure);
  mSoundSpeed = dataBase.newFluidFieldList(0.0, HydroFieldNames::soundSpeed);
  mOmegaGradh = dataBase.newFluidFieldList(1.0, HydroFieldNames::omegaGradh);
  mSpecificThermalEnergy0 = dataBase.newFluidFieldList(0.0, HydroFieldNames::specificThermalEnergy + "0");
  mEntropy = dataBase.newFluidFieldList(0.0, HydroFieldNames::entropy);
  mHideal = dataBase.newFluidFieldList(SymTensor::zero, ReplaceBoundedState<Dimension, Field<Dimension, SymTensor> >::prefix() + HydroFieldNames::H);
  mMaxViscousPressure = dataBase.newFluidFieldList(0.0, HydroFieldNames::maxViscousPressure);
  mEffViscousPressure = dataBase.newFluidFieldList(0.0, HydroFieldNames::effectiveViscousPressure);
  mMassDensityCorrection = dataBase.newFluidFieldList(0.0, HydroFieldNames::massDensityCorrection);
  mViscousWork = dataBase.newFluidFieldList(0.0, HydroFieldNames::viscousWork);
  mMassDensitySum = dataBase.newFluidFieldList(0.0, ReplaceFieldList<Dimension, Field<Dimension, SymTensor> >::prefix() + HydroFieldNames::massDensity);
  mNormalization = dataBase.newFluidFieldList(0.0, HydroFieldNames::normalization);
  mWeightedNeighborSum = dataBase.newFluidFieldList(0.0, HydroFieldNames::weightedNeighborSum);
  mMassSecondMoment = dataBase.newFluidFieldList(SymTensor::zero, HydroFieldNames::massSecondMoment);
  mXSPHWeightSum = dataBase.newFluidFieldList(0.0, HydroFieldNames::XSPHWeightSum);
  mXSPHDeltaV = dataBase.newFluidFieldList(Vector::zero, HydroFieldNames::XSPHDeltaV);
  mDxDt = dataBase.newFluidFieldList(Vector::zero, IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position);
  mDvDt = dataBase.newFluidFieldList(Vector::zero, HydroFieldNames::hydroAcceleration);
  mDmassDensityDt = dataBase.newFluidFieldList(0.0, IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity);
  mDspecificThermalEnergyDt = dataBase.newFluidFieldList(0.0, IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy);
  mDHDt = dataBase.newFluidFieldList(SymTensor::zero, IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::H);
  mDvDx = dataBase.newFluidFieldList(Tensor::zero, HydroFieldNames::velocityGradient);
  mInternalDvDx = dataBase.newFluidFieldList(Tensor::zero, HydroFieldNames::internalVelocityGradient);
  mPairAccelerations.clear();
  mM = dataBase.newFluidFieldList(Tensor::zero, HydroFieldNames::M_SPHCorrection);
  mLocalM = dataBase.newFluidFieldList(Tensor::zero, "local " + HydroFieldNames::M_SPHCorrection);
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
SPHHydroBase<Dimension>::
~SPHHydroBase() {
}

//------------------------------------------------------------------------------
// On problem start up, we need to initialize our internal data.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
initializeProblemStartup(DataBase<Dimension>& dataBase) {

  TIME_SPHinitializeStartup.start();
  // Initialize the pressure and sound speed.
  dataBase.fluidPressure(mPressure);
  dataBase.fluidSoundSpeed(mSoundSpeed);
  // dataBase.fluidEntropy(mEntropy);

  // // In some cases we need the volume per node as well.
  // const bool updateVolume = (this->densityUpdate() == VoronoiCellDensity or
  //                            this->densityUpdate() == SumVoronoiCellDensity);
  // if (updateVolume) {
  //   Mesh<Dimension> mesh;
  //   NodeList<Dimension> voidNodes("void", 0, 0);
  //   vector<const NodeList<Dimension>*> nodeLists(dataBase.nodeListBegin(), dataBase.nodeListEnd());
  //   nodeLists.push_back(&voidNodes);
  //   sort(nodeLists.begin(), nodeLists.end(), typename NodeListRegistrar<Dimension>::NodeListComparator());
  //   Vector xmin, xmax;
  //   boundingBox(dataBase.fluidPosition(), xmin, xmax);
  //   generateMesh<Dimension, 
  //                           typename vector<const NodeList<Dimension>*>::iterator,
  //                           ConstBoundaryIterator>(nodeLists.begin(), nodeLists.end(),
  //                                                  this->boundaryBegin(), this->boundaryEnd(),
  //                                                  xmin, xmax, 
  //                                                  true,           // generateVoid
  //                                                  false,          // generateParallelConnectivity
  //                                                  false,          // remove boundary zones
  //                                                  2.0,            // voidThreshold
  //                                                  mesh,
  //                                                  voidNodes);

  //   mVolume = dataBase.newFluidFieldList(0.0, HydroFieldNames::volume);
  //   for (unsigned nodeListi = 0; nodeListi != dataBase.numFluidNodeLists(); ++nodeListi) {
  //     const unsigned n = mVolume[nodeListi]->numInternalElements();
  //     const unsigned offset = mesh.offset(nodeListi);
  //     for (unsigned i = 0; i != n; ++i) {
  //       mVolume(nodeListi, i) = mesh.zone(offset + i).volume();
  //     }
  //   }
  // }
  TIME_SPHinitializeStartup.stop();
}

//------------------------------------------------------------------------------
// Register the state we need/are going to evolve.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
registerState(DataBase<Dimension>& dataBase,
              State<Dimension>& state) {
  TIME_SPHregister.start();
  typedef typename State<Dimension>::PolicyPointer PolicyPointer;

  // Create the local storage for time step mask, pressure, sound speed, and position weight.
  dataBase.resizeFluidFieldList(mTimeStepMask, 1, HydroFieldNames::timeStepMask);
  // dataBase.fluidPressure(mPressure);
  // dataBase.fluidSoundSpeed(mSoundSpeed);
  dataBase.resizeFluidFieldList(mOmegaGradh, 1.0, HydroFieldNames::omegaGradh);

  // We may need the volume per node as well.
  const bool updateVolume = (this->densityUpdate() == MassDensityType::VoronoiCellDensity or
                             this->densityUpdate() == MassDensityType::SumVoronoiCellDensity);
  if (updateVolume) {
    dataBase.resizeFluidFieldList(mVolume, 0.0, HydroFieldNames::volume, false);
  }

  // We have to choose either compatible or total energy evolution.
  VERIFY2(not (mCompatibleEnergyEvolution and mEvolveTotalEnergy),
          "SPH error : you cannot simultaneously use both compatibleEnergyEvolution and evolveTotalEnergy");

  // If we're using the compatibile energy discretization, prepare to maintain a copy
  // of the thermal energy.
  dataBase.resizeFluidFieldList(mSpecificThermalEnergy0, 0.0);
  // dataBase.resizeFluidFieldList(mEntropy, 0.0, HydroFieldNames::entropy, false);
  if (mCompatibleEnergyEvolution) {
    size_t nodeListi = 0;
    for (typename DataBase<Dimension>::FluidNodeListIterator itr = dataBase.fluidNodeListBegin();
         itr != dataBase.fluidNodeListEnd();
         ++itr, ++nodeListi) {
      *mSpecificThermalEnergy0[nodeListi] = (*itr)->specificThermalEnergy();
      (*mSpecificThermalEnergy0[nodeListi]).name(HydroFieldNames::specificThermalEnergy + "0");
    }
  }

  // Now register away.
  // Mass.
  FieldList<Dimension, Scalar> mass = dataBase.fluidMass();
  state.enroll(mass);

  // We need to build up CompositeFieldListPolicies for the mass density and H fields
  // in order to enforce NodeList dependent limits.
  FieldList<Dimension, Scalar> massDensity = dataBase.fluidMassDensity();
  FieldList<Dimension, SymTensor> Hfield = dataBase.fluidHfield();
  std::shared_ptr<CompositeFieldListPolicy<Dimension, Scalar> > rhoPolicy(new CompositeFieldListPolicy<Dimension, Scalar>());
  std::shared_ptr<CompositeFieldListPolicy<Dimension, SymTensor> > Hpolicy(new CompositeFieldListPolicy<Dimension, SymTensor>());
  for (typename DataBase<Dimension>::FluidNodeListIterator itr = dataBase.fluidNodeListBegin();
       itr != dataBase.fluidNodeListEnd();
       ++itr) {
    rhoPolicy->push_back(new IncrementBoundedState<Dimension, Scalar>((*itr)->rhoMin(),
                                                                      (*itr)->rhoMax()));
    const Scalar hmaxInv = 1.0/(*itr)->hmax();
    const Scalar hminInv = 1.0/(*itr)->hmin();
    if (HEvolution() == HEvolutionType::IntegrateH) {
      Hpolicy->push_back(new IncrementBoundedState<Dimension, SymTensor, Scalar>(hmaxInv, hminInv));
    } else {
      CHECK(HEvolution() == HEvolutionType::IdealH);
      Hpolicy->push_back(new ReplaceBoundedState<Dimension, SymTensor, Scalar>(hmaxInv, hminInv));
    }
  }
  state.enroll(massDensity, rhoPolicy);
  state.enroll(Hfield, Hpolicy);

  // Volume.
  if (updateVolume) state.enroll(mVolume);

  // Register the position update.
  FieldList<Dimension, Vector> position = dataBase.fluidPosition();
  if (true) { // (mXSPH) {
    PolicyPointer positionPolicy(new IncrementFieldList<Dimension, Vector>());
    state.enroll(position, positionPolicy);
  } else {
    PolicyPointer positionPolicy(new PositionPolicy<Dimension>());
    state.enroll(position, positionPolicy);
  }

  // Are we using the compatible energy evolution scheme?
  FieldList<Dimension, Scalar> specificThermalEnergy = dataBase.fluidSpecificThermalEnergy();
  FieldList<Dimension, Vector> velocity = dataBase.fluidVelocity();
  if (mCompatibleEnergyEvolution) {
    PolicyPointer thermalEnergyPolicy(new SpecificThermalEnergyPolicy<Dimension>(dataBase));
    PolicyPointer velocityPolicy(new IncrementFieldList<Dimension, Vector>(HydroFieldNames::position,
                                                                           HydroFieldNames::specificThermalEnergy,
                                                                           true));
    state.enroll(specificThermalEnergy, thermalEnergyPolicy);
    state.enroll(velocity, velocityPolicy);
    state.enroll(mSpecificThermalEnergy0);

  } else if (mEvolveTotalEnergy) {
    // If we're doing total energy, we register the specific energy to advance with the
    // total energy policy.
    PolicyPointer thermalEnergyPolicy(new SpecificFromTotalThermalEnergyPolicy<Dimension>());
    PolicyPointer velocityPolicy(new IncrementFieldList<Dimension, Vector>(HydroFieldNames::position,
                                                                           HydroFieldNames::specificThermalEnergy,
                                                                           true));
    state.enroll(specificThermalEnergy, thermalEnergyPolicy);
    state.enroll(velocity, velocityPolicy);

  } else {
    // Otherwise we're just time-evolving the specific energy.
    PolicyPointer thermalEnergyPolicy(new IncrementFieldList<Dimension, Scalar>());
    PolicyPointer velocityPolicy(new IncrementFieldList<Dimension, Vector>(HydroFieldNames::position,
                                                                           true));
    state.enroll(specificThermalEnergy, thermalEnergyPolicy);
    state.enroll(velocity, velocityPolicy);
  }

  // Register the time step mask, initialized to 1 so that everything defaults to being
  // checked.
  state.enroll(mTimeStepMask);

  // Compute and register the pressure and sound speed.
  PolicyPointer pressurePolicy(new PressurePolicy<Dimension>());
  PolicyPointer csPolicy(new SoundSpeedPolicy<Dimension>());
  state.enroll(mPressure, pressurePolicy);
  state.enroll(mSoundSpeed, csPolicy);

  // Register the grad h correction terms
  // We deliberately make this non-dynamic here.  These corrections are computed
  // during SPHHydroBase::initialize, not as part of our usual state update.
  state.enroll(mOmegaGradh);
  TIME_SPHregister.stop();
}

//------------------------------------------------------------------------------
// Register the state derivative fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
registerDerivatives(DataBase<Dimension>& dataBase,
                    StateDerivatives<Dimension>& derivs) {
  TIME_SPHregisterDerivs.start();

  // Create the scratch fields.
  // Note we deliberately do not zero out the derivatives here!  This is because the previous step
  // info here may be used by other algorithms (like the CheapSynchronousRK2 integrator or
  // the ArtificialVisocisity::initialize step).
  dataBase.resizeFluidFieldList(mHideal, SymTensor::zero, ReplaceBoundedState<Dimension, Field<Dimension, SymTensor> >::prefix() + HydroFieldNames::H, false);
  dataBase.resizeFluidFieldList(mMaxViscousPressure, 0.0, HydroFieldNames::maxViscousPressure, false);
  dataBase.resizeFluidFieldList(mEffViscousPressure, 0.0, HydroFieldNames::effectiveViscousPressure, false);
  dataBase.resizeFluidFieldList(mMassDensityCorrection, 0.0, HydroFieldNames::massDensityCorrection, false);
  dataBase.resizeFluidFieldList(mViscousWork, 0.0, HydroFieldNames::viscousWork, false);
  dataBase.resizeFluidFieldList(mMassDensitySum, 0.0, ReplaceFieldList<Dimension, Field<Dimension, SymTensor> >::prefix() + HydroFieldNames::massDensity, false);
  dataBase.resizeFluidFieldList(mNormalization, 0.0, HydroFieldNames::normalization, false);
  dataBase.resizeFluidFieldList(mWeightedNeighborSum, 0.0, HydroFieldNames::weightedNeighborSum, false);
  dataBase.resizeFluidFieldList(mMassSecondMoment, SymTensor::zero, HydroFieldNames::massSecondMoment, false);
  dataBase.resizeFluidFieldList(mXSPHWeightSum, 0.0, HydroFieldNames::XSPHWeightSum, false);
  dataBase.resizeFluidFieldList(mXSPHDeltaV, Vector::zero, HydroFieldNames::XSPHDeltaV, false);
  dataBase.resizeFluidFieldList(mDvDt, Vector::zero, HydroFieldNames::hydroAcceleration, false);
  dataBase.resizeFluidFieldList(mDmassDensityDt, 0.0, IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, false);
  dataBase.resizeFluidFieldList(mDspecificThermalEnergyDt, 0.0, IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy, false);
  dataBase.resizeFluidFieldList(mDHDt, SymTensor::zero, IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::H, false);
  dataBase.resizeFluidFieldList(mDvDx, Tensor::zero, HydroFieldNames::velocityGradient, false);
  dataBase.resizeFluidFieldList(mInternalDvDx, Tensor::zero, HydroFieldNames::internalVelocityGradient, false);
  dataBase.resizeFluidFieldList(mM, Tensor::zero, HydroFieldNames::M_SPHCorrection, false);
  dataBase.resizeFluidFieldList(mLocalM, Tensor::zero, "local " + HydroFieldNames::M_SPHCorrection, false);
  derivs.enroll(mHideal);
  derivs.enroll(mMaxViscousPressure);
  derivs.enroll(mEffViscousPressure);
  derivs.enroll(mMassDensityCorrection);
  derivs.enroll(mViscousWork);
  derivs.enroll(mMassDensitySum);
  derivs.enroll(mNormalization);
  derivs.enroll(mWeightedNeighborSum);
  derivs.enroll(mMassSecondMoment);
  derivs.enroll(mXSPHWeightSum);
  derivs.enroll(mXSPHDeltaV);

  // Check if someone already registered DxDt.
  if (not derivs.registered(mDxDt)) {
    dataBase.resizeFluidFieldList(mDxDt, Vector::zero, IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, false);
    derivs.enroll(mDxDt);
  }

  // Check that no-one else is trying to control the hydro vote for DvDt.
  CHECK(not derivs.registered(mDvDt));
  derivs.enroll(mDvDt);

  derivs.enroll(mDmassDensityDt);
  derivs.enroll(mDspecificThermalEnergyDt);
  derivs.enroll(mDHDt);
  derivs.enroll(mDvDx);
  derivs.enroll(mInternalDvDx);
  derivs.enroll(mM);
  derivs.enroll(mLocalM);
  derivs.enrollAny(HydroFieldNames::pairAccelerations, mPairAccelerations);
  TIME_SPHregisterDerivs.stop();
}

//------------------------------------------------------------------------------
// This method is called once at the beginning of a timestep, after all state registration.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
preStepInitialize(const DataBase<Dimension>& dataBase, 
                  State<Dimension>& state,
                  StateDerivatives<Dimension>& derivs) {
  TIME_SPHpreStepInitialize.start();

  // Depending on the mass density advancement selected, we may want to replace the 
  // mass density.
  switch(densityUpdate()) {

  case MassDensityType::RigorousSumDensity:
  case MassDensityType::CorrectedSumDensity:
    {
      const auto& connectivityMap = dataBase.connectivityMap();
      const auto  position = state.fields(HydroFieldNames::position, Vector::zero);
      const auto  mass = state.fields(HydroFieldNames::mass, 0.0);
      const auto  H = state.fields(HydroFieldNames::H, SymTensor::zero);
      auto        massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
      computeSPHSumMassDensity(connectivityMap, this->kernel(), mSumMassDensityOverAllNodeLists, position, mass, H, massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
      if (densityUpdate() == MassDensityType::CorrectedSumDensity) {
        correctSPHSumMassDensity(connectivityMap, this->kernel(), mSumMassDensityOverAllNodeLists, position, mass, H, massDensity);
        for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
        for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
      }
    }
    break;

  case MassDensityType::SumDensity:
    {
      auto       massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
      const auto massDensitySum = derivs.fields(ReplaceFieldList<Dimension, Field<Dimension, Field<Dimension, Scalar> > >::prefix() + 
                                                HydroFieldNames::massDensity, 0.0);
      massDensity.assignFields(massDensitySum);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
    }
    break;

  case MassDensityType::HybridSumDensity:
    {
      auto       massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
      const auto massDensitySum = derivs.fields(ReplaceFieldList<Dimension, Field<Dimension, Field<Dimension, Scalar> > >::prefix() + 
                                                HydroFieldNames::massDensity, 0.0);
      const auto normalization = state.fields(HydroFieldNames::normalization, 0.0);
      const auto numNodeLists = normalization.size();
      for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
        const auto n = normalization[nodeListi]->numInternalElements();
        for (auto i = 0u; i < n; ++i) {
          if (normalization(nodeListi, i) > 0.95) massDensity(nodeListi, i) = massDensitySum(nodeListi, i);
        }
      }
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
    }
    break;

  case MassDensityType::VoronoiCellDensity:
    {
      this->updateVolume(state, false);
      const auto mass = state.fields(HydroFieldNames::mass, 0.0);
      const auto volume = state.fields(HydroFieldNames::volume, 0.0);
      auto       massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
      massDensity = mass / volume;
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
    }
    break;

  case MassDensityType::SumVoronoiCellDensity:
    {
      this->updateVolume(state, true);
      const auto& connectivityMap = dataBase.connectivityMap();
      const auto  position = state.fields(HydroFieldNames::position, Vector::zero);
      const auto  mass = state.fields(HydroFieldNames::mass, 0.0);
      const auto  volume = state.fields(HydroFieldNames::volume, 0.0);
      const auto  H = state.fields(HydroFieldNames::H, SymTensor::zero);
      auto        massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
      computeSumVoronoiCellMassDensity(connectivityMap, this->kernel(), position, mass, volume, H, massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
    }
    break;
  default:
    break;
  }

  // // This form looks for points that are too close based on specific volume.
  // if (mfilter > 0.0) {
  //   const TableKernel<Dimension>& W = this->kernel();
  //   const ConnectivityMap<Dimension>& connectivityMap = dataBase.connectivityMap();
  //   FieldList<Dimension, Vector> position = state.fields(HydroFieldNames::position, Vector::zero);
  //   const FieldList<Dimension, Scalar> mass = state.fields(HydroFieldNames::mass, 0.0);
  //   const FieldList<Dimension, Vector> velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  //   const FieldList<Dimension, SymTensor> H = state.fields(HydroFieldNames::H, SymTensor::zero);
  //   const FieldList<Dimension, Scalar> massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  //   const unsigned numNodeLists = mass.size();
  //   const Scalar W0 = W.kernelValue(0.0, 1.0);
  //   FieldList<Dimension, Vector> deltar = dataBase.newFluidFieldList(Vector::zero, "delta position");
  //   FieldList<Dimension, Scalar> deltav = dataBase.newFluidFieldList(0.0, "delta velocity");
  //   FieldList<Dimension, Scalar> weightsum = dataBase.newFluidFieldList(0.0, "delta velocity weight sum");
  //   for (unsigned nodeListi = 0; nodeListi != numNodeLists; ++nodeListi) {
  //     const Scalar nPerh = position[nodeListi]->nodeListPtr()->nodesPerSmoothingScale();
  //     for (typename ConnectivityMap<Dimension>::const_iterator iItr = connectivityMap.begin(nodeListi);
  //          iItr != connectivityMap.end(nodeListi);
  //          ++iItr) {
  //       const int i = *iItr;
  //       const Vector& ri = position(nodeListi, i);
  //       const Vector& vi = velocity(nodeListi, i);
  //       const Scalar mi = mass(nodeListi, i);
  //       const Scalar rhoi = massDensity(nodeListi, i);
  //       const SymTensor& Hi = H(nodeListi, i);
  //       const SymTensor Hinvi = Hi.Inverse();
  //       const vector<vector<int> >& fullConnectivity = connectivityMap.connectivityForNode(nodeListi, i);
  //       for (unsigned nodeListj = 0; nodeListj != numNodeLists; ++nodeListj) {
  //         for (typename vector<int>::const_iterator jItr = fullConnectivity[nodeListj].begin();
  //              jItr != fullConnectivity[nodeListj].end();
  //              ++jItr) {
  //           const unsigned j = *jItr;
  //           const Vector& rj = position(nodeListj, j);
  //           const Vector& vj = velocity(nodeListj, j);
  //           const Scalar mj = mass(nodeListj, j);
  //           const Scalar rhoj = massDensity(nodeListj, j);
  //           const SymTensor& Hj = H(nodeListj, j);
  //           const SymTensor Hinvj = Hj.Inverse();
  //           const Vector rji = rj - ri;
  //           const Vector rjihat = rji.unitVector();
  //           const Vector etai = Hi*rji;
  //           const Vector etaj = Hj*rji;
  //           const Scalar etaMagi = etai.magnitude();
  //           const Scalar etaMagj = etaj.magnitude();
  //           const Vector delta = 0.5*(max(0.0, 1.0/nPerh - etaMagi)*Hinvi + max(0.0, 1.0/nPerh - etaMagj)*Hinvj)*rjihat;
  //           const Scalar weight = 0.5*(W.kernelValue(etaMagi, 1.0) + W.kernelValue(etaMagj, 1.0))/W0 * (vj - vi).magnitude();
  //           deltar(nodeListi, i) -= weight*delta;
  //           weightsum(nodeListi, i) += weight;
  //           deltav(nodeListi, i) += weight*(vj - vi).magnitude();
  //         }
  //       }
  //     }
  //   }

  //   // Apply the filtering.
  //   const FieldList<Dimension, Vector> DxDt = derivs.fields(IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, Vector::zero);
  //   for (unsigned nodeListi = 0; nodeListi != numNodeLists; ++nodeListi) {
  //     const unsigned n = position[nodeListi]->numInternalElements();
  //     for (unsigned i = 0; i != n; ++i) {
  //       // const Scalar hi = 1.0/(H(nodeListi, i).eigenValues().maxElement());
  //       // const Scalar mag0 = DvDx(nodeListi, i).eigenValues().maxAbsElement()*hi*dt;
  //       // const Scalar mag0 = DxDt(nodeListi, i).magnitude() * dt;
  //       const Scalar mag0 = deltav(nodeListi, i)*safeInv(weightsum(nodeListi, i))*dt;
  //       if (mag0 > 0.0) {
  //         const Scalar deltamag = deltar(nodeListi, i).magnitude();
  //         // const Scalar effmag = mfilter*deltamag;
  //         const Scalar effmag = mfilter*min(mag0, deltamag);
  //         position(nodeListi, i) += effmag*deltar(nodeListi, i).unitVector();
  //       }
  //     }
  //   }

  //   // Check for any boundary violations.
  //   for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
  //        boundaryItr != this->boundaryEnd();
  //        ++boundaryItr) (*boundaryItr)->setAllViolationNodes(dataBase);
  //   this->enforceBoundaries(state, derivs);
  // }
  TIME_SPHpreStepInitialize.stop();
}

//------------------------------------------------------------------------------
// Initialize the hydro before calling evaluateDerivatives
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
initialize(const typename Dimension::Scalar time,
           const typename Dimension::Scalar dt,
           const DataBase<Dimension>& dataBase,
           State<Dimension>& state,
           StateDerivatives<Dimension>& derivs) {
  TIME_SPHinitialize.start();

  // Initialize the grad h corrrections if needed.
  //const TableKernel<Dimension>& W = this->kernel();
  const TableKernel<Dimension>& WPi = this->PiKernel();

  if (mGradhCorrection) {
    const ConnectivityMap<Dimension>& connectivityMap = dataBase.connectivityMap();
    const FieldList<Dimension, Vector> position = state.fields(HydroFieldNames::position, Vector::zero);
    const FieldList<Dimension, SymTensor> H = state.fields(HydroFieldNames::H, SymTensor::zero);
    FieldList<Dimension, Scalar> omega = state.fields(HydroFieldNames::omegaGradh, 0.0);
    computeSPHOmegaGradhCorrection(connectivityMap, this->kernel(), position, H, omega);
    for (ConstBoundaryIterator boundItr = this->boundaryBegin();
         boundItr != this->boundaryEnd();
         ++boundItr) (*boundItr)->applyFieldListGhostBoundary(omega);
  }

  // Get the artificial viscosity and initialize it.
  ArtificialViscosity<Dimension>& Q = this->artificialViscosity();
  Q.initialize(dataBase, 
               state,
               derivs,
               this->boundaryBegin(),
               this->boundaryEnd(),
               time, 
               dt,
               WPi);

  // We depend on the caller knowing to finalize the ghost boundaries!
  TIME_SPHinitialize.stop();
}

//------------------------------------------------------------------------------
// Finalize the derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
finalizeDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar /*dt*/,
                    const DataBase<Dimension>& /*dataBase*/,
                    const State<Dimension>& /*state*/,
                    StateDerivatives<Dimension>& derivs) const {
  TIME_SPHfinalizeDerivs.start();

  // If we're using the compatible energy discretization, we need to enforce
  // boundary conditions on the accelerations.
  if (compatibleEnergyEvolution()) {
    auto accelerations = derivs.fields(HydroFieldNames::hydroAcceleration, Vector::zero);
    auto DepsDt = derivs.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy, 0.0);
    for (ConstBoundaryIterator boundaryItr = this->boundaryBegin();
         boundaryItr != this->boundaryEnd();
         ++boundaryItr) {
      (*boundaryItr)->applyFieldListGhostBoundary(accelerations);
      (*boundaryItr)->applyFieldListGhostBoundary(DepsDt);
    }
    for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
         boundaryItr != this->boundaryEnd();
         ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
  }
  TIME_SPHfinalizeDerivs.stop();
}

//------------------------------------------------------------------------------
// Apply the ghost boundary conditions for hydro state fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
applyGhostBoundaries(State<Dimension>& state,
                     StateDerivatives<Dimension>& /*derivs*/) {
  TIME_SPHghostBounds.start();

  // Apply boundary conditions to the basic fluid state Fields.
  FieldList<Dimension, Scalar> mass = state.fields(HydroFieldNames::mass, 0.0);
  FieldList<Dimension, Scalar> massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  FieldList<Dimension, Scalar> specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  FieldList<Dimension, Vector> velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  FieldList<Dimension, Scalar> pressure = state.fields(HydroFieldNames::pressure, 0.0);
  FieldList<Dimension, Scalar> soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);
  FieldList<Dimension, Scalar> omega = state.fields(HydroFieldNames::omegaGradh, 0.0);
  FieldList<Dimension, Scalar> specificThermalEnergy0;
  if (compatibleEnergyEvolution()) {
    CHECK(state.fieldNameRegistered(HydroFieldNames::specificThermalEnergy + "0"));
    specificThermalEnergy0 = state.fields(HydroFieldNames::specificThermalEnergy + "0", 0.0);
  }

  // FieldList<Dimension, Scalar> volume;
  // const bool updateVolume = (this->densityUpdate() == MassDensityType::VoronoiCellDensity or
  //                            this->densityUpdate() == MassDensityType::SumVoronoiCellDensity);
  // if (updateVolume) {
  //   CHECK(state.fieldNameRegistered(HydroFieldNames::volume));
  //   volume = state.fields(HydroFieldNames::volume, 0.0);
  // }

  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
    (*boundaryItr)->applyFieldListGhostBoundary(mass);
    (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
    (*boundaryItr)->applyFieldListGhostBoundary(specificThermalEnergy);
    (*boundaryItr)->applyFieldListGhostBoundary(velocity);
    (*boundaryItr)->applyFieldListGhostBoundary(pressure);
    (*boundaryItr)->applyFieldListGhostBoundary(soundSpeed);
    (*boundaryItr)->applyFieldListGhostBoundary(omega);
    if (compatibleEnergyEvolution()) {
      (*boundaryItr)->applyFieldListGhostBoundary(specificThermalEnergy0);
    }
    // if (updateVolume) (*boundaryItr)->applyFieldListGhostBoundary(volume);
  }
  TIME_SPHghostBounds.stop();
}

//------------------------------------------------------------------------------
// Enforce the boundary conditions for hydro state fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
enforceBoundaries(State<Dimension>& state,
                  StateDerivatives<Dimension>& /*derivs*/) {
  TIME_SPHenforceBounds.start();

  // Enforce boundary conditions on the fluid state Fields.
  FieldList<Dimension, Scalar> mass = state.fields(HydroFieldNames::mass, 0.0);
  FieldList<Dimension, Scalar> massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  FieldList<Dimension, Scalar> specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  FieldList<Dimension, Vector> velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  FieldList<Dimension, Scalar> pressure = state.fields(HydroFieldNames::pressure, 0.0);
  FieldList<Dimension, Scalar> soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);
  FieldList<Dimension, Scalar> omega = state.fields(HydroFieldNames::omegaGradh, 0.0);

  FieldList<Dimension, Scalar> specificThermalEnergy0;
  if (compatibleEnergyEvolution()) {
    specificThermalEnergy0 = state.fields(HydroFieldNames::specificThermalEnergy + "0", 0.0);
  }

  // FieldList<Dimension, Scalar> volume;
  // const bool updateVolume = (this->densityUpdate() == MassDensityType::VoronoiCellDensity or
  //                            this->densityUpdate() == MassDensityType::SumVoronoiCellDensity);
  // if (updateVolume) {
  //   CHECK(state.fieldNameRegistered(HydroFieldNames::volume));
  //   volume = state.fields(HydroFieldNames::volume, 0.0);
  // }

  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
    (*boundaryItr)->enforceFieldListBoundary(mass);
    (*boundaryItr)->enforceFieldListBoundary(massDensity);
    (*boundaryItr)->enforceFieldListBoundary(specificThermalEnergy);
    (*boundaryItr)->enforceFieldListBoundary(velocity);
    (*boundaryItr)->enforceFieldListBoundary(pressure);
    (*boundaryItr)->enforceFieldListBoundary(soundSpeed);
    (*boundaryItr)->enforceFieldListBoundary(omega);
    if (compatibleEnergyEvolution()) {
      (*boundaryItr)->enforceFieldListBoundary(specificThermalEnergy0);
    }
    // if (updateVolume) (*boundaryItr)->enforceFieldListBoundary(volume);
  }
  TIME_SPHenforceBounds.stop();
}

//------------------------------------------------------------------------------
// Update the volume field in the State.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
updateVolume(State<Dimension>& state,
             const bool boundaries) const {
  TIME_SPHupdateVol.start();

  // Pre-conditions.
  REQUIRE(state.fieldNameRegistered(HydroFieldNames::position));
  REQUIRE(state.fieldNameRegistered(HydroFieldNames::volume));
  REQUIRE(state.meshRegistered());

  // Find the global bounding box.
  Vector xmin, xmax;
  const FieldList<Dimension, Vector> positions = state.fields(HydroFieldNames::position, Vector::zero);
  globalBoundingBox<Dimension>(positions, xmin, xmax, 
                               false);     // ghost points

  // Puff things up a bit.
  const Vector delta = 0.1*(xmax - xmin);
  xmin -= delta;
  xmax += delta;

  // Create the mesh.
  Mesh<Dimension>& mesh = state.mesh();
  mesh.clear();
  NodeList<Dimension> voidNodes("void", 0, 0);
  vector<const NodeList<Dimension>*> nodeLists(positions.nodeListPtrs().begin(),
                                               positions.nodeListPtrs().end());
  nodeLists.push_back(&voidNodes);
  generateMesh<Dimension, 
                          typename vector<const NodeList<Dimension>*>::iterator,
                          ConstBoundaryIterator>
    (nodeLists.begin(), nodeLists.end(),
     this->boundaryBegin(),
     this->boundaryEnd(),
     xmin, xmax,
     true,                             // meshGhostNodes
     false,                            // generateVoid
     false,                            // generateParallelConnectivity
     false,                            // removeBoundaryZones
     2.0,                              // voidThreshold
     mesh,
     voidNodes);

  // Extract the volume.
  FieldList<Dimension, Scalar> volume = state.fields(HydroFieldNames::volume, 0.0);

  // Now walk the NodeLists and set the volume.
  const unsigned numNodeLists = volume.size();
  unsigned nodeListi, i, offset, numInternal;
  for (nodeListi = 0; nodeListi != numNodeLists; ++nodeListi) {
    offset = mesh.offset(nodeListi);
    numInternal = volume[nodeListi]->numInternalElements();
    for (i = 0; i != numInternal; ++i) {
      volume(nodeListi, i) = mesh.zone(i + offset).volume();
    }
    std::fill(volume[nodeListi]->begin() + numInternal,
         volume[nodeListi]->end(),
         1.0e-10);
  }

  // Optionally fill in the boundary values for the volume.
  if (boundaries) {
    for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
         boundaryItr != this->boundaryEnd();
         ++boundaryItr) (*boundaryItr)->applyFieldListGhostBoundary(volume);
    for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
         boundaryItr != this->boundaryEnd();
         ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
  }

  // That's it.
  TIME_SPHupdateVol.stop();
}

//------------------------------------------------------------------------------
// Dump the current state to the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
dumpState(FileIO& file, const string& pathName) const {
  file.write(mTimeStepMask, pathName + "/timeStepMask");
  file.write(mPressure, pathName + "/pressure");
  file.write(mSoundSpeed, pathName + "/soundSpeed");
  file.write(mVolume, pathName + "/volume");
  file.write(mSpecificThermalEnergy0, pathName + "/specificThermalEnergy0");
  // file.write(mEntropy, pathName + "/entropy");
  file.write(mHideal, pathName + "/Hideal");
  file.write(mMassDensitySum, pathName + "/massDensitySum");
  file.write(mNormalization, pathName + "/normalization");
  file.write(mWeightedNeighborSum, pathName + "/weightedNeighborSum");
  file.write(mMassSecondMoment, pathName + "/massSecondMoment");
  file.write(mXSPHWeightSum, pathName + "/XSPHWeightSum");
  file.write(mXSPHDeltaV, pathName + "/XSPHDeltaV");

  file.write(mOmegaGradh, pathName + "/omegaGradh");
  file.write(mDxDt, pathName + "/DxDt");
  file.write(mDvDt, pathName + "/DvDt");
  file.write(mDmassDensityDt, pathName + "/DmassDensityDt");
  file.write(mDspecificThermalEnergyDt, pathName + "/DspecificThermalEnergyDt");
  file.write(mDHDt, pathName + "/DHDt");
  file.write(mDvDx, pathName + "/DvDx");
  file.write(mInternalDvDx, pathName + "/internalDvDx");
  file.write(mMaxViscousPressure, pathName + "/maxViscousPressure");
  file.write(mEffViscousPressure, pathName + "/effectiveViscousPressure");
  file.write(mMassDensityCorrection, pathName + "/massDensityCorrection");
  file.write(mViscousWork, pathName + "/viscousWork");
  file.write(mM, pathName + "/M");
  file.write(mLocalM, pathName + "/localM");

//   this->artificialViscosity().dumpState(file, pathName + "/Q");

}

//------------------------------------------------------------------------------
// Restore the state from the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
restoreState(const FileIO& file, const string& pathName) {
 
  file.read(mTimeStepMask, pathName + "/timeStepMask");
  file.read(mPressure, pathName + "/pressure");
  file.read(mSoundSpeed, pathName + "/soundSpeed");
  file.read(mVolume, pathName + "/volume");
  file.read(mSpecificThermalEnergy0, pathName + "/specificThermalEnergy0");
  // file.read(mEntropy, pathName + "/entropy");
  file.read(mHideal, pathName + "/Hideal");
  file.read(mMassDensitySum, pathName + "/massDensitySum");
  file.read(mNormalization, pathName + "/normalization");
  file.read(mWeightedNeighborSum, pathName + "/weightedNeighborSum");
  file.read(mMassSecondMoment, pathName + "/massSecondMoment");
  file.read(mXSPHWeightSum, pathName + "/XSPHWeightSum");
  file.read(mXSPHDeltaV, pathName + "/XSPHDeltaV");
  file.read(mOmegaGradh, pathName + "/omegaGradh");
  file.read(mDxDt, pathName + "/DxDt");
  file.read(mDvDt, pathName + "/DvDt");
  file.read(mDmassDensityDt, pathName + "/DmassDensityDt");
  file.read(mDspecificThermalEnergyDt, pathName + "/DspecificThermalEnergyDt");
  file.read(mDHDt, pathName + "/DHDt");
  file.read(mDvDx, pathName + "/DvDx");
  file.read(mInternalDvDx, pathName + "/internalDvDx");
  file.read(mMaxViscousPressure, pathName + "/maxViscousPressure");
  file.read(mM, pathName + "/M");
  file.read(mLocalM, pathName + "/localM");

  // For backwards compatibility on change 3597 -- drop in the near future.
  for (auto DvDtPtr: mDvDt) DvDtPtr->name(HydroFieldNames::hydroAcceleration);

//   this->artificialViscosity().restoreState(file, pathName + "/Q");
}

}
