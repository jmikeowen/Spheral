//---------------------------------Spheral++----------------------------------//
// RSPHHydroBase -- The SPH/ASPH hydrodynamic package for Spheral++.
//
// Created by JMO, Mon Jul 19 22:11:09 PDT 2010
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "NodeList/SmoothingScaleBase.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Physics/Physics.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/IncrementFieldList.hh"
#include "DataBase/ReplaceFieldList.hh"
#include "DataBase/IncrementBoundedFieldList.hh"
#include "DataBase/ReplaceBoundedFieldList.hh"
#include "DataBase/IncrementBoundedState.hh"
#include "DataBase/ReplaceBoundedState.hh"
#include "DataBase/CompositeFieldListPolicy.hh"
//#include "Hydro/VolumePolicy.hh"
//#include "Hydro/VoronoiMassDensityPolicy.hh"
//#include "Hydro/SumVoronoiMassDensityPolicy.hh"
#include "Hydro/SpecificThermalEnergyPolicy.hh"
#include "Hydro/SpecificFromTotalThermalEnergyPolicy.hh"
#include "Hydro/PositionPolicy.hh"
#include "Hydro/PressurePolicy.hh"
#include "Hydro/SoundSpeedPolicy.hh"
//#include "Hydro/EntropyPolicy.hh"
//#include "Mesh/MeshPolicy.hh"
//#include "Mesh/generateMesh.hh"
#include "DataBase/DataBase.hh"
#include "Field/FieldList.hh"
#include "Field/NodeIterators.hh"
#include "Boundary/Boundary.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Utilities/timingUtilities.hh"
#include "Utilities/safeInv.hh"
#include "Utilities/globalBoundingVolumes.hh"
#include "Utilities/Timer.hh"
//#include "Mesh/Mesh.hh"
//#include "CRKSPH/volumeSpacing.hh"

#include "RSPHHydroBase.hh"

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
extern Timer TIME_RSPH;
extern Timer TIME_RSPHinitializeStartup;
extern Timer TIME_RSPHregister;
extern Timer TIME_RSPHregisterDerivs;
extern Timer TIME_RSPHpreStepInitialize;
extern Timer TIME_RSPHinitialize;
extern Timer TIME_RSPHfinalizeDerivs;
extern Timer TIME_RSPHghostBounds;
extern Timer TIME_RSPHupdateVol;
extern Timer TIME_RSPHenforceBounds;
extern Timer TIME_RSPHevalDerivs;
extern Timer TIME_RSPHevalDerivs_initial;
extern Timer TIME_RSPHevalDerivs_pairs;
extern Timer TIME_RSPHevalDerivs_final;

namespace Spheral {

//------------------------------------------------------------------------------
// Construct with the given artificial viscosity and kernels.
//------------------------------------------------------------------------------
template<typename Dimension>
RSPHHydroBase<Dimension>::
RSPHHydroBase(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
             DataBase<Dimension>& dataBase,
             const TableKernel<Dimension>& W,
             const double cfl,
             const bool useVelocityMagnitudeForDt,
             const bool compatibleEnergyEvolution,
             const bool evolveTotalEnergy,
             const bool XSPH,
             const bool correctVelocityGradient,
             const HEvolutionType HUpdate,
             const double epsTensile,
             const double nTensile,
             const Vector& xmin,
             const Vector& xmax):
  Physics<Dimension>(),
  mCfl(cfl),
  mUseVelocityMagnitudeForDt(useVelocityMagnitudeForDt),
  mMinMasterNeighbor(INT_MAX),
  mMaxMasterNeighbor(0),
  mSumMasterNeighbor(0),
  mMinCoarseNeighbor(INT_MAX),
  mMaxCoarseNeighbor(0),
  mSumCoarseNeighbor(0),
  mMinRefineNeighbor(INT_MAX),
  mMaxRefineNeighbor(0),
  mSumRefineNeighbor(0),
  mMinActualNeighbor(INT_MAX),
  mMaxActualNeighbor(0),
  mSumActualNeighbor(0),
  mNormMasterNeighbor(0),
  mNormCoarseNeighbor(0),
  mNormRefineNeighbor(0),
  mNormActualNeighbor(0),
  mKernel(W),
  mSmoothingScaleMethod(smoothingScaleMethod),
  mHEvolution(HUpdate),
  mCompatibleEnergyEvolution(compatibleEnergyEvolution),
  mEvolveTotalEnergy(evolveTotalEnergy),
  mXSPH(XSPH),
  mCorrectVelocityGradient(correctVelocityGradient),
  mEpsTensile(epsTensile),
  mnTensile(nTensile),
  mxmin(xmin),
  mxmax(xmax),
  mTimeStepMask(FieldStorageType::CopyFields),
  mPressure(FieldStorageType::CopyFields),
  mSoundSpeed(FieldStorageType::CopyFields),
  mHideal(FieldStorageType::CopyFields),
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
  mPairAccelerations(),
  mDpDx(FieldStorageType::CopyFields),
  mLastDvDx(FieldStorageType::CopyFields),
  mRestart(registerWithRestart(*this)) {

  // Create storage for our internal state.
  mTimeStepMask = dataBase.newFluidFieldList(int(0), HydroFieldNames::timeStepMask);
  mPressure = dataBase.newFluidFieldList(0.0, HydroFieldNames::pressure);
  mSoundSpeed = dataBase.newFluidFieldList(0.0, HydroFieldNames::soundSpeed);
  mHideal = dataBase.newFluidFieldList(SymTensor::zero, ReplaceBoundedState<Dimension, Field<Dimension, SymTensor> >::prefix() + HydroFieldNames::H);
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
  mDpDx = dataBase.newFluidFieldList(Vector::zero, "pressureGradient");
  mLastDvDx = dataBase.newFluidFieldList(Tensor::zero, "velocityGradientLastTimeStep");
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
RSPHHydroBase<Dimension>::
~RSPHHydroBase() {
}

//------------------------------------------------------------------------------
// On problem start up, we need to initialize our internal data.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
initializeProblemStartup(DataBase<Dimension>& dataBase) {

  TIME_RSPHinitializeStartup.start();

  dataBase.fluidPressure(mPressure);
  dataBase.fluidSoundSpeed(mSoundSpeed);
  
  TIME_RSPHinitializeStartup.stop();
}

//------------------------------------------------------------------------------
// Register the state we need/are going to evolve.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
registerState(DataBase<Dimension>& dataBase,
              State<Dimension>& state) {
  TIME_RSPHregister.start();
  typedef typename State<Dimension>::PolicyPointer PolicyPointer;
  // We have to choose either compatible or total energy evolution.
  VERIFY2(not (mCompatibleEnergyEvolution and mEvolveTotalEnergy),
          "SPH error : you cannot simultaneously use both compatibleEnergyEvolution and evolveTotalEnergy");

  // Create the local storage for time step mask, pressure, sound speed, and position weight.
  dataBase.resizeFluidFieldList(mTimeStepMask, 1, HydroFieldNames::timeStepMask);
  dataBase.resizeFluidFieldList(mLastDvDx, Tensor::zero, "velocityGradientLastTimeStep" ,false);

  FieldList<Dimension, Scalar> mass = dataBase.fluidMass();
  FieldList<Dimension, Scalar> massDensity = dataBase.fluidMassDensity();
  FieldList<Dimension, SymTensor> Hfield = dataBase.fluidHfield();
  FieldList<Dimension, Vector> position = dataBase.fluidPosition();
  FieldList<Dimension, Scalar> specificThermalEnergy = dataBase.fluidSpecificThermalEnergy();
  FieldList<Dimension, Vector> velocity = dataBase.fluidVelocity();

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
  
  PolicyPointer positionPolicy(new IncrementFieldList<Dimension, Vector>());
  PolicyPointer pressurePolicy(new PressurePolicy<Dimension>());
  PolicyPointer csPolicy(new SoundSpeedPolicy<Dimension>());
  
  state.enroll(mTimeStepMask);
  state.enroll(mass);
  state.enroll(mLastDvDx);
  state.enroll(massDensity, rhoPolicy);
  state.enroll(Hfield, Hpolicy);
  state.enroll(position, positionPolicy);
  state.enroll(mPressure, pressurePolicy);
  state.enroll(mSoundSpeed, csPolicy);


  // conditional for energy method
  if (mEvolveTotalEnergy) {
    PolicyPointer thermalEnergyPolicy(new SpecificFromTotalThermalEnergyPolicy<Dimension>());
    PolicyPointer velocityPolicy(new IncrementFieldList<Dimension, Vector>(HydroFieldNames::position,
                                                                           HydroFieldNames::specificThermalEnergy,
                                                                           true));
    state.enroll(specificThermalEnergy, thermalEnergyPolicy);
    state.enroll(velocity, velocityPolicy);

  } else {
    PolicyPointer thermalEnergyPolicy(new IncrementFieldList<Dimension, Scalar>());
    PolicyPointer velocityPolicy(new IncrementFieldList<Dimension, Vector>(HydroFieldNames::position,
                                                                           true));
    state.enroll(specificThermalEnergy, thermalEnergyPolicy);
    state.enroll(velocity, velocityPolicy);
  }

 
  TIME_RSPHregister.stop();
}

//------------------------------------------------------------------------------
// Register the state derivative fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
registerDerivatives(DataBase<Dimension>& dataBase,
                    StateDerivatives<Dimension>& derivs) {
  TIME_RSPHregisterDerivs.start();

  // Create the scratch fields.
  // Note we deliberately do not zero out the derivatives here!  This is because the previous step
  // info here may be used by other algorithms (like the CheapSynchronousRK2 integrator or
  dataBase.resizeFluidFieldList(mDpDx, Vector::zero,"pressureGradient", false);
  dataBase.resizeFluidFieldList(mHideal, SymTensor::zero, ReplaceBoundedState<Dimension, Field<Dimension, SymTensor> >::prefix() + HydroFieldNames::H, false);
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

  // Check if someone already registered DxDt.
  if (not derivs.registered(mDxDt)) {
    dataBase.resizeFluidFieldList(mDxDt, Vector::zero, IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, false);
    derivs.enroll(mDxDt);
  }
  // Check that no-one else is trying to control the hydro vote for DvDt.
  CHECK(not derivs.registered(mDvDt));
  derivs.enroll(mDpDx);
  derivs.enroll(mDvDt);
  derivs.enroll(mHideal);
  derivs.enroll(mNormalization);
  derivs.enroll(mWeightedNeighborSum);
  derivs.enroll(mMassSecondMoment);
  derivs.enroll(mXSPHWeightSum);
  derivs.enroll(mXSPHDeltaV);
  derivs.enroll(mDmassDensityDt);
  derivs.enroll(mDspecificThermalEnergyDt);
  derivs.enroll(mDHDt);
  derivs.enroll(mDvDx);
  derivs.enroll(mInternalDvDx);
  derivs.enroll(mM);
  derivs.enroll(mLocalM);
  derivs.enrollAny(HydroFieldNames::pairAccelerations, mPairAccelerations);
  TIME_RSPHregisterDerivs.stop();
}

//------------------------------------------------------------------------------
// RSPH Time step
//------------------------------------------------------------------------------
template<typename Dimension>
typename RSPHHydroBase<Dimension>::TimeStepType
RSPHHydroBase<Dimension>::
dt(const DataBase<Dimension>& dataBase,
   const State<Dimension>& state,
   const StateDerivatives<Dimension>& derivs,
   const Scalar currentTime) const{

}
//------------------------------------------------------------------------------
// This method is called once at the beginning of a timestep, after all state registration.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
preStepInitialize(const DataBase<Dimension>& /*dataBase*/, 
                  State<Dimension>& /*state*/,
                  StateDerivatives<Dimension>& /*derivs*/) {
  TIME_RSPHpreStepInitialize.start();

  TIME_RSPHpreStepInitialize.stop();
}

//------------------------------------------------------------------------------
// Initialize the hydro before calling evaluateDerivatives
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
initialize(const typename Dimension::Scalar /*time*/,
           const typename Dimension::Scalar /*dt*/,
           const DataBase<Dimension>& /*dataBase*/,
           State<Dimension>& state,
           StateDerivatives<Dimension>& derivs) {
  TIME_RSPHinitialize.start();

  // store DvDx for use in HLLC reconstruction
  const auto DvDx = derivs.fields(HydroFieldNames::velocityGradient, Tensor::zero);
  auto DvDx0 = state.fields("velocityGradientLastTimeStep",Tensor::zero);
  DvDx0.copyFields();

  for (ConstBoundaryIterator boundItr = this->boundaryBegin();
         boundItr != this->boundaryEnd();
         ++boundItr)(*boundItr)->applyFieldListGhostBoundary(DvDx0);
  
  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
         boundaryItr != this->boundaryEnd();
         ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
 
  TIME_RSPHinitialize.stop();
}


//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
evaluateDerivatives(const typename Dimension::Scalar time,
                    const typename Dimension::Scalar dt,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivatives) const {
  TIME_RSPHevalDerivs.start();
  TIME_RSPHevalDerivs_initial.start();

  this->evaluateSpatialGradients(time,dt,dataBase,state,derivatives);

  // The kernels and such.
  const auto& W = this->kernel();

  // A few useful constants we'll use in the following loop.
  const double tiny = 1.0e-30;
  const Scalar W0 = W(0.0, 1.0);

  // The connectivity.
  const auto& connectivityMap = dataBase.connectivityMap();
  const auto& nodeLists = connectivityMap.nodeLists();
  const auto numNodeLists = nodeLists.size();

  // Get the state and derivative FieldLists.
  // State FieldLists.
  const auto mass = state.fields(HydroFieldNames::mass, 0.0);
  const auto position = state.fields(HydroFieldNames::position, Vector::zero);
  const auto velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  const auto massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  const auto H = state.fields(HydroFieldNames::H, SymTensor::zero);
  const auto pressure = state.fields(HydroFieldNames::pressure, 0.0);
  const auto soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);
  const auto oldDvDx = state.fields("velocityGradientLastTimeStep",Tensor::zero);
  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(velocity.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);
  CHECK(pressure.size() == numNodeLists);
  CHECK(soundSpeed.size() == numNodeLists);
  CHECK(oldDvDx.size() == numNodeLists);

  // Derivative FieldLists.
  auto  normalization = derivatives.fields(HydroFieldNames::normalization, 0.0);
  auto  DxDt = derivatives.fields(IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, Vector::zero);
  auto  DrhoDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto  DvDt = derivatives.fields(HydroFieldNames::hydroAcceleration, Vector::zero);
  auto  DepsDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy, 0.0);
  auto  DvDx = derivatives.fields(HydroFieldNames::velocityGradient, Tensor::zero);
  auto  localDvDx = derivatives.fields(HydroFieldNames::internalVelocityGradient, Tensor::zero);
  auto  DHDt = derivatives.fields(IncrementFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto  Hideal = derivatives.fields(ReplaceBoundedFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto& pairAccelerations = derivatives.getAny(HydroFieldNames::pairAccelerations, vector<Vector>());
  auto  XSPHWeightSum = derivatives.fields(HydroFieldNames::XSPHWeightSum, 0.0);
  auto  XSPHDeltaV = derivatives.fields(HydroFieldNames::XSPHDeltaV, Vector::zero);
  auto  weightedNeighborSum = derivatives.fields(HydroFieldNames::weightedNeighborSum, 0.0);
  auto  massSecondMoment = derivatives.fields(HydroFieldNames::massSecondMoment, SymTensor::zero);
  const auto DpDx = derivatives.fields("pressureGradient",Vector::zero);
  CHECK(normalization.size() == numNodeLists);
  CHECK(DxDt.size() == numNodeLists);
  CHECK(DrhoDt.size() == numNodeLists);
  CHECK(DvDt.size() == numNodeLists);
  CHECK(DepsDt.size() == numNodeLists);
  CHECK(DvDx.size() == numNodeLists);
  CHECK(localDvDx.size() == numNodeLists);
  CHECK(DHDt.size() == numNodeLists);
  CHECK(Hideal.size() == numNodeLists);
  CHECK(XSPHWeightSum.size() == numNodeLists);
  CHECK(XSPHDeltaV.size() == numNodeLists);
  CHECK(weightedNeighborSum.size() == numNodeLists);
  CHECK(massSecondMoment.size() == numNodeLists);
  CHECK(DpDx.size() == numNodeLists);

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();

  // Size up the pair-wise accelerations before we start.
  if (mCompatibleEnergyEvolution) pairAccelerations.resize(npairs);

  // The scale for the tensile correction.
  const auto& nodeList = mass[0]->nodeList();
  const auto  nPerh = nodeList.nodesPerSmoothingScale();
  const auto  WnPerh = W(1.0/nPerh, 1.0);
  TIME_RSPHevalDerivs_initial.stop();

  // Walk all the interacting pairs.
  TIME_RSPHevalDerivs_pairs.start();
#pragma omp parallel
  {
    // Thread private scratch variables
    int i, j, nodeListi, nodeListj;
    Scalar Wi, gWi, Wj, gWj, Pstar;
    Vector vstar;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;
    auto normalization_thread = normalization.threadCopy(threadStack);
    auto DvDt_thread = DvDt.threadCopy(threadStack);
    auto DepsDt_thread = DepsDt.threadCopy(threadStack);
    auto DvDx_thread = DvDx.threadCopy(threadStack);
    auto localDvDx_thread = localDvDx.threadCopy(threadStack);
    auto XSPHWeightSum_thread = XSPHWeightSum.threadCopy(threadStack);
    auto XSPHDeltaV_thread = XSPHDeltaV.threadCopy(threadStack);
    auto weightedNeighborSum_thread = weightedNeighborSum.threadCopy(threadStack);
    auto massSecondMoment_thread = massSecondMoment.threadCopy(threadStack);

#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto& ci = soundSpeed(nodeListi, i);
      const auto& DpDxi = DpDx(nodeListi,i);
      const auto& oldDvDxi = oldDvDx(nodeListi,i);
      const auto  Hdeti = Hi.Determinant();
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& normi = normalization_thread(nodeListi, i);
      auto& DvDti = DvDt_thread(nodeListi, i);
      auto& DepsDti = DepsDt_thread(nodeListi, i);
      auto& DvDxi = DvDx_thread(nodeListi, i);
      auto& localDvDxi = localDvDx_thread(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum_thread(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV_thread(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum_thread(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment_thread(nodeListi, i);

      // Get the state for node j
      const auto& rj = position(nodeListj, j);
      const auto& mj = mass(nodeListj, j);
      const auto& vj = velocity(nodeListj, j);
      const auto& rhoj = massDensity(nodeListj, j);
      const auto& Pj = pressure(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto& cj = soundSpeed(nodeListj, j);
      const auto& DpDxj = DpDx(nodeListj,j);
      const auto& oldDvDxj = oldDvDx(nodeListj,j);
      const auto  Hdetj = Hj.Determinant();
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& normj = normalization_thread(nodeListj, j);
      auto& DvDtj = DvDt_thread(nodeListj, j);
      auto& DepsDtj = DepsDt_thread(nodeListj, j);
      auto& DvDxj = DvDx_thread(nodeListj, j);
      auto& localDvDxj = localDvDx_thread(nodeListj, j);
      auto& XSPHWeightSumj = XSPHWeightSum_thread(nodeListj, j);
      auto& XSPHDeltaVj = XSPHDeltaV_thread(nodeListj, j);
      auto& weightedNeighborSumj = weightedNeighborSum_thread(nodeListj, j);
      auto& massSecondMomentj = massSecondMoment_thread(nodeListj, j);

      // Flag if this is a contiguous material pair or not.
      const bool sameMatij =  (nodeListi == nodeListj);

      // Node displacement.
      const auto rij = ri - rj;
      const auto vij = vi - vj;
      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);

      // Symmetrized kernel weight and gradient.
      std::tie(Wi, gWi) = W.kernelAndGradValue(etaMagi, Hdeti);
      const auto Hetai = Hi*etai.unitVector();
      const auto gradWi = gWi*Hetai;

      std::tie(Wj, gWj) = W.kernelAndGradValue(etaMagj, Hdetj);
      const auto Hetaj = Hj*etaj.unitVector();
      const auto gradWj = gWj*Hetaj;

      // Zero'th and second moment of the node distribution -- used for the
      // ideal H calculation.
      const auto rij2 = rij.magnitude2();
      const auto thpt = rij.selfdyad()*safeInvVar(rij2*rij2*rij2);
      weightedNeighborSumi += std::abs(gWi);
      weightedNeighborSumj += std::abs(gWj);
      massSecondMomenti += gradWi.magnitude2()*thpt;
      massSecondMomentj += gradWj.magnitude2()*thpt;


      // Determine an effective pressure including a term to fight the tensile instability.
//             const auto fij = epsTensile*pow(Wi/(Hdeti*WnPerh), nTensile);
      const auto fij = mEpsTensile*FastMath::pow4(Wi/(Hdeti*WnPerh));
      const auto Ri = fij*(Pi < 0.0 ? -Pi : 0.0);
      const auto Rj = fij*(Pj < 0.0 ? -Pj : 0.0);
      const auto Peffi = Pi + Ri;
      const auto Peffj = Pj + Rj;


      // averaged things.
      const auto rhoij = 0.5*(rhoi+rhoj); 
      const auto cij = 0.5*(ci+cj);  
      const auto Wij = 0.5*(Wi+Wj); 
      const auto gWij = 0.5*(gWi+gWj);
      const auto gradWij = 0.5*(gradWi+gradWj);

      // Acceleration.
      CHECK(rhoi > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(ci > 0.0);
      CHECK(cj > 0.0);

      const auto  voli = mi/rhoi;
      const auto  volj = mj/rhoj;

      const auto isExpanding = vij.dot(rij) > 0.0;

      if (isExpanding){
        vstar = 0.5*(vi+vj);
        Pstar = 0.5*(Pi+Pj);
      }else{
        computeHLLCstate( rij, 
                        nodeListi, nodeListj, i, j,
                        Peffi, Peffj, rhoi, rhoj, vi, vj, ci, cj,
                        DpDxi, DpDxj, oldDvDxi, oldDvDxj,
                        vstar, Pstar);
      }

      const auto Prho = Pstar/(rhoi*rhoj)*(gradWij);
      const auto deltaDvDt = 2*Prho;
      DvDti -= mj*deltaDvDt;
      DvDtj += mi*deltaDvDt;
      if (mCompatibleEnergyEvolution) pairAccelerations[kk] = -mj*deltaDvDt;  // Acceleration for i (j anti-symmetric)

      DepsDti += mj*(Prho.dot(vi-vstar));
      DepsDtj += mi*(Prho.dot(vstar-vj));


      // Velocity gradient.
      const auto deltaDvDxi = volj*(vi-vstar).dyad(gradWij);
      const auto deltaDvDxj = voli*(vstar-vj).dyad(gradWij);
      DvDxi -= deltaDvDxi; 
      DvDxj -= deltaDvDxj;
      if (sameMatij) {
        localDvDxi -= deltaDvDxi; 
        localDvDxj -= deltaDvDxj;
      }

    } // loop over pairs

    // Reduce the thread values to the master.
    threadReduceFieldLists<Dimension>(threadStack);

  }   // OpenMP parallel region
  TIME_RSPHevalDerivs_pairs.stop();

  // Finish up the derivatives for each point.
  TIME_RSPHevalDerivs_final.start();
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto& nodeList = mass[nodeListi]->nodeList();
    const auto  hmin = nodeList.hmin();
    const auto  hmax = nodeList.hmax();
    const auto  hminratio = nodeList.hminratio();
    const auto  nPerh = nodeList.nodesPerSmoothingScale();

    const auto ni = nodeList.numInternalNodes();
#pragma omp parallel for
    for (auto i = 0u; i < ni; ++i) {

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& normi = normalization(nodeListi, i);
      auto& DxDti = DxDt(nodeListi, i);
      auto& DrhoDti = DrhoDt(nodeListi, i);
      auto& DvDti = DvDt(nodeListi, i);
      auto& DepsDti = DepsDt(nodeListi, i);
      auto& DvDxi = DvDx(nodeListi, i);
      auto& localDvDxi = localDvDx(nodeListi, i);
      auto& DHDti = DHDt(nodeListi, i);
      auto& Hideali = Hideal(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment(nodeListi, i);

      // Add the self-contribution to density sum.
      normi += mi/rhoi*W0*Hdeti;

      // Evaluate the continuity equation.
      DrhoDti = -rhoi*DvDxi.Trace();

      // If needed finish the total energy derivative.
      if (mEvolveTotalEnergy) DepsDti = mi*(vi.dot(DvDti) + DepsDti);

      // Complete the moments of the node distribution for use in the ideal H calculation.
      weightedNeighborSumi = Dimension::rootnu(max(0.0, weightedNeighborSumi/Hdeti));
      massSecondMomenti /= Hdeti*Hdeti;

      // Determine the position evolution, based on whether we're doing XSPH or not.
      DxDti = vi;
      //if (mXSPH) {
      //  XSPHWeightSumi += Hdeti*mi/rhoi*W0;
      //  CHECK2(XSPHWeightSumi != 0.0, i << " " << XSPHWeightSumi);
      //  DxDti += XSPHDeltaVi/max(tiny, XSPHWeightSumi);
      //} 

      // The H tensor evolution.
      DHDti = mSmoothingScaleMethod.smoothingScaleDerivative(Hi,
                                                             ri,
                                                             DvDxi,
                                                             hmin,
                                                             hmax,
                                                             hminratio,
                                                             nPerh);
      Hideali = mSmoothingScaleMethod.newSmoothingScale(Hi,
                                                        ri,
                                                        weightedNeighborSumi,
                                                        massSecondMomenti,
                                                        W,
                                                        hmin,
                                                        hmax,
                                                        hminratio,
                                                        nPerh,
                                                        connectivityMap,
                                                        nodeListi,
                                                        i);
    }
  }
  TIME_RSPHevalDerivs_final.stop();
  TIME_RSPHevalDerivs.stop();
}


//------------------------------------------------------------------------------
// EvalDerivs subroutine for spatial derivs
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
evaluateSpatialGradients(const typename Dimension::Scalar /*time*/,
                         const typename Dimension::Scalar /*dt*/,
                         const DataBase<Dimension>& dataBase,
                         const State<Dimension>& state,
                               StateDerivatives<Dimension>& derivatives) const {


  // The kernels and such.
  const auto& W = this->kernel();

  // A few useful constants we'll use in the following loop.
  const double tiny = 1.0e-30;
  const Scalar W0 = W(0.0, 1.0);

  // The connectivity.
  const auto& connectivityMap = dataBase.connectivityMap();
  const auto& nodeLists = connectivityMap.nodeLists();
  const auto numNodeLists = nodeLists.size();

  // Get the state and derivative FieldLists.
  // State FieldLists.
  const auto mass = state.fields(HydroFieldNames::mass, 0.0);
  const auto position = state.fields(HydroFieldNames::position, Vector::zero);
  const auto massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  const auto pressure = state.fields(HydroFieldNames::pressure, 0.0);
  const auto H = state.fields(HydroFieldNames::H, SymTensor::zero);

  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);

  // Derivative FieldLists.
  auto  M = derivatives.fields(HydroFieldNames::M_SPHCorrection, Tensor::zero);
  auto DpDx = derivatives.fields("pressureGradient", Vector::zero)

  CHECK(M.size() == numNodeLists);
  CHECK(DpDx.size() == numNodeLists);

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();

  // The scale for the tensile correction.
  const auto& nodeList = mass[0]->nodeList();
  const auto  nPerh = nodeList.nodesPerSmoothingScale();
  const auto  WnPerh = W(1.0/nPerh, 1.0);
  TIME_RSPHevalDerivs_initial.stop();

  // Walk all the interacting pairs.
  TIME_RSPHevalDerivs_pairs.start();
#pragma omp parallel
  {
    // Thread private scratch variables
    int i, j, nodeListi, nodeListj;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;

    auto M_thread = M.threadCopy(threadStack);
    auto DpDx_thread = DpDx.threadCopy(threadStack);

#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& Mi = M_thread(nodeListi, i);
      auto& DpDxi = DpDx_thread(nodeListi,i);

      // Get the state for node j
      const auto& rj = position(nodeListj, j);
      const auto& mj = mass(nodeListj, j);
      const auto& rhoj = massDensity(nodeListj, j);
      const auto& Pj = pressure(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto  Hdetj = Hj.Determinant();
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& Mj = M_thread(nodeListj, j);
      auto& DpDxj = DpDx_thread(nodeListj,j);

      // Flag if this is a contiguous material pair or not.
      const bool sameMatij =  (nodeListi == nodeListj);

      // Node displacement.
      const auto rij = ri - rj;

      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);

      // Symmetrized kernel weight and gradient.
      const auto gWi = W.gradValue(etaMagi, Hdeti);
      const auto Hetai = Hi*etai.unitVector();
      const auto gradWi = gWi*Hetai;

      const auto gWj = W.gradValue(etaMagj, Hdetj);
      const auto Hetaj = Hj*etaj.unitVector();
      const auto gradWj = gWj*Hetaj;

      // averaged things.
      const auto gradWij = 0.5*(gradWi+gradWj);
      
      // volumes
      const auto voli = mi/rhoi;
      const auto volj = mj/rhoj;

      const Vector DpDxij = (Pi-Pj) * gradWij;

      DpDxi -= volj * DpDxij;
      DpDxj -= voli * DpDxij;

      // Linear gradient correction term.
      if(this->mCorrectVelocityGradient){
        const auto Mij = rij.dyad(gradWij);
        Mi -= volj*Mij;
        Mj -= voli*Mij;
      }

    } // loop over pairs

    // Reduce the thread values to the master.
    threadReduceFieldLists<Dimension>(threadStack);

  }   // OpenMP parallel region

  // Finish up the spatial gradient calculation
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto& nodeList = mass[nodeListi]->nodeList();
    const auto ni = nodeList.numInternalNodes();
#pragma omp parallel for
    for (auto i = 0u; i < ni; ++i) {
      if (this->mCorrectVelocityGradient){
        const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
        auto& Mi = M(nodeListi, i);
        auto& DpDxi = DpDx(nodeListi, i);
        const auto goodM =  (this->mCorrectVelocityGradient and
                             std::abs(Mi.Determinant()) > 1.0e-10 and
                             numNeighborsi > Dimension::pownu(2));
        Mi = (goodM? Mi.Inverse() : Tensor::one);
        DpDxi = Mi.Transpose()*DpDxi;
      }

    }
  }

  for (ConstBoundaryIterator boundItr = this->boundaryBegin();
         boundItr != this->boundaryEnd();
         ++boundItr){
    (*boundItr)->applyFieldListGhostBoundary(DpDx);
    if(this->mCorrectVelocityGradient)(*boundItr)->applyFieldListGhostBoundary(M);
  }
  
  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
         boundaryItr != this->boundaryEnd();
         ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
 

}
//------------------------------------------------------------------------------
// Finalize the derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
finalizeDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar /*dt*/,
                    const DataBase<Dimension>& /*dataBase*/,
                    const State<Dimension>& /*state*/,
                    StateDerivatives<Dimension>& /*derivs*/) const {
  TIME_RSPHfinalizeDerivs.start();

  TIME_RSPHfinalizeDerivs.stop();
}

//------------------------------------------------------------------------------
// Apply the ghost boundary conditions for hydro state fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
applyGhostBoundaries(State<Dimension>& state,
                     StateDerivatives<Dimension>& /*derivs*/) {
  TIME_RSPHghostBounds.start();

  // Apply boundary conditions to the basic fluid state Fields.
  FieldList<Dimension, Scalar> mass = state.fields(HydroFieldNames::mass, 0.0);
  FieldList<Dimension, Scalar> massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  FieldList<Dimension, Scalar> specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  FieldList<Dimension, Vector> velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  FieldList<Dimension, Scalar> pressure = state.fields(HydroFieldNames::pressure, 0.0);
  FieldList<Dimension, Scalar> soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);


  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
    (*boundaryItr)->applyFieldListGhostBoundary(mass);
    (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
    (*boundaryItr)->applyFieldListGhostBoundary(specificThermalEnergy);
    (*boundaryItr)->applyFieldListGhostBoundary(velocity);
    (*boundaryItr)->applyFieldListGhostBoundary(pressure);
    (*boundaryItr)->applyFieldListGhostBoundary(soundSpeed);
  }
  TIME_RSPHghostBounds.stop();
}

//------------------------------------------------------------------------------
// Enforce the boundary conditions for hydro state fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
enforceBoundaries(State<Dimension>& state,
                  StateDerivatives<Dimension>& /*derivs*/) {
  TIME_RSPHenforceBounds.start();

  // Enforce boundary conditions on the fluid state Fields.
  FieldList<Dimension, Scalar> mass = state.fields(HydroFieldNames::mass, 0.0);
  FieldList<Dimension, Scalar> massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  FieldList<Dimension, Scalar> specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  FieldList<Dimension, Vector> velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  FieldList<Dimension, Scalar> pressure = state.fields(HydroFieldNames::pressure, 0.0);
  FieldList<Dimension, Scalar> soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);


  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
    (*boundaryItr)->enforceFieldListBoundary(mass);
    (*boundaryItr)->enforceFieldListBoundary(massDensity);
    (*boundaryItr)->enforceFieldListBoundary(specificThermalEnergy);
    (*boundaryItr)->enforceFieldListBoundary(velocity);
    (*boundaryItr)->enforceFieldListBoundary(pressure);
    (*boundaryItr)->enforceFieldListBoundary(soundSpeed);
  }
  TIME_RSPHenforceBounds.stop();
}


//------------------------------------------------------------------------------
// Dump the current state to the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
dumpState(FileIO& file, const string& pathName) const {
  file.write(mTimeStepMask, pathName + "/timeStepMask");
  file.write(mPressure, pathName + "/pressure");
  file.write(mSoundSpeed, pathName + "/soundSpeed");
  file.write(mHideal, pathName + "/Hideal");
  file.write(mNormalization, pathName + "/normalization");
  file.write(mWeightedNeighborSum, pathName + "/weightedNeighborSum");
  file.write(mMassSecondMoment, pathName + "/massSecondMoment");
  file.write(mXSPHWeightSum, pathName + "/XSPHWeightSum");
  file.write(mXSPHDeltaV, pathName + "/XSPHDeltaV");


  file.write(mDxDt, pathName + "/DxDt");
  file.write(mDvDt, pathName + "/DvDt");
  file.write(mDmassDensityDt, pathName + "/DmassDensityDt");
  file.write(mDspecificThermalEnergyDt, pathName + "/DspecificThermalEnergyDt");
  file.write(mDHDt, pathName + "/DHDt");
  file.write(mDvDx, pathName + "/DvDx");
  file.write(mInternalDvDx, pathName + "/internalDvDx");
  file.write(mM, pathName + "/M");
  file.write(mLocalM, pathName + "/localM");

//   this->artificialViscosity().dumpState(file, pathName + "/Q");

}

//------------------------------------------------------------------------------
// Restore the state from the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
RSPHHydroBase<Dimension>::
restoreState(const FileIO& file, const string& pathName) {
 
  file.read(mTimeStepMask, pathName + "/timeStepMask");
  file.read(mPressure, pathName + "/pressure");
  file.read(mSoundSpeed, pathName + "/soundSpeed");
  file.read(mHideal, pathName + "/Hideal");
  file.read(mNormalization, pathName + "/normalization");
  file.read(mWeightedNeighborSum, pathName + "/weightedNeighborSum");
  file.read(mMassSecondMoment, pathName + "/massSecondMoment");
  file.read(mXSPHWeightSum, pathName + "/XSPHWeightSum");
  file.read(mXSPHDeltaV, pathName + "/XSPHDeltaV");
  file.read(mDxDt, pathName + "/DxDt");
  file.read(mDvDt, pathName + "/DvDt");
  file.read(mDmassDensityDt, pathName + "/DmassDensityDt");
  file.read(mDspecificThermalEnergyDt, pathName + "/DspecificThermalEnergyDt");
  file.read(mDHDt, pathName + "/DHDt");
  file.read(mDvDx, pathName + "/DvDx");
  file.read(mInternalDvDx, pathName + "/internalDvDx");
  file.read(mM, pathName + "/M");
  file.read(mLocalM, pathName + "/localM");

  // For backwards compatibility on change 3597 -- drop in the near future.
  for (auto DvDtPtr: mDvDt) DvDtPtr->name(HydroFieldNames::hydroAcceleration);

}




template<typename Dimension>
void
RSPHHydroBase<Dimension>::
computeHLLCstate( const typename Dimension::Vector& rij,
                  int nodeListi,
                  int nodeListj,
                  int i,
                  int j,
                  const typename Dimension::Scalar& Pi,
                  const typename Dimension::Scalar& Pj,
                  const typename Dimension::Scalar& rhoi, 
                  const typename Dimension::Scalar& rhoj,
                  const typename Dimension::Vector& vi,   
                  const typename Dimension::Vector& vj,
                  const typename Dimension::Scalar& ci,   
                  const typename Dimension::Scalar& cj,
                  const typename Dimension::Vector& DpDxi,
                  const typename Dimension::Vector& DpDxj,
                  const typename Dimension::Tensor& DvDxi,
                  const typename Dimension::Tensor& DvDxj,
                  typename Dimension::Vector& vstar,     
                  typename Dimension::Scalar& Pstar) const {



  // normal component
  const auto rhatij = rij.unitVector();
  //const auto isExpanding = (vi-vj).dot(rij) > 0.0;

  // perpendicular component 
  //const auto wij = vi+vj-(ui+uj)*rhatij;                      
  //const auto wi = vi-ui*rhatij;
  //const auto wj = vj-ui*rhatij;
  
  // dumb 1D wave speed 
  //const auto Si = ui + ci;
  //const auto Sj = uj - cj;

  // accoustic impedance
  //const auto Ci = rhoi*(Si-ui);
  //const auto Cj = rhoj*(Sj-uj);

  // get our vanLeer limiter 
  const auto xij = rij/2.0;

  //auto gradi = DrhoDxi.dot(xij);
  //auto gradj = DrhoDxj.dot(xij);
  //auto ri = gradi/(sgn(gradj)*max(1.0e-30, abs(gradj)));
  //auto rj = gradj/(sgn(gradi)*max(1.0e-30, abs(gradi)));
  //auto x = min(ri,rj);

  //auto phi = ( x>0.0 ? 
  //             2.0/(1.0 + x)*2.0*x/(1.0 + x) : 
  //             0.0
  //           );
  
  // project linear soln
  //const auto rhoR = rhoi - phi*DrhoDxi.dot(xij);
  //const auto rhoL = rhoj + phi*DrhoDxj.dot(xij);

  //gradi = DcDxi.dot(xij);
  //gradj = DcDxj.dot(xij);
  //ri = gradi/(sgn(gradj)*max(1.0e-30, abs(gradj)));
  //rj = gradj/(sgn(gradi)*max(1.0e-30, abs(gradi)));
  //x = min(ri,rj);

  //phi = ( x>0.0 ? 
  //        2.0/(1.0 + x)*2.0*x/(1.0 + x) : 
  //        0.0
  //      );
  
  // project linear soln
  //const auto cR = ci - phi*DcDxi.dot(xij);
  //const auto cL = cj + phi*DcDxj.dot(xij);

  //gradi = DpDxi.dot(xij);
  //gradj = DpDxj.dot(xij);
  //ri = gradi/(sgn(gradj)*max(1.0e-30, abs(gradj)));
  //rj = gradj/(sgn(gradi)*max(1.0e-30, abs(gradi)));
  //x = min(ri,rj);

  //phi = ( x>0.0 ? 
  //        2.0/(1.0 + x)*2.0*x/(1.0 + x) : 
  //        0.0
  //      );
  
  // project linear soln
  //const auto pR = Pi - phi*DpDxi.dot(xij);
  //const auto pL = Pj + phi*DpDxj.dot(xij);


  const auto gradi = (DvDxi.dot(xij)).dot(xij);
  const auto gradj = (DvDxj.dot(xij)).dot(xij);
  const auto ri = gradi/(sgn(gradj)*max(1.0e-30, abs(gradj)));
  const auto rj = gradj/(sgn(gradi)*max(1.0e-30, abs(gradi)));
  const auto x = min(ri,rj);

  const auto phi = ( x>0.0 ? 
                     2.0/(1.0 + x)*2.0*x/(1.0 + x) : 
                     0.0
                   );
  
  // project linear soln
  const auto vi1 = vi - phi*DvDxi*xij;
  const auto vj1 = vj + phi*DvDxj*xij;
  const auto ui = vi1.dot(rhatij);
  const auto uj = vj1.dot(rhatij);

    // dont go out of bounds
  //const auto ui = vi.dot(rhatij);
  //const auto uj = vj.dot(rhatij);
  const auto umin = min(ui,uj);
  const auto umax = max(ui,uj);

  // 1D interface velocity 
  const auto ustar = ((ci > 0.0 and cj>0.0) ?
                      min(max((rhoi*ci*ui + rhoj*cj*uj - Pi + Pj )/max((rhoi*ci + rhoj*cj),1e-30),umin),umax) :
                      (ui+uj)/2.0);

  vstar = ustar*rhatij + ((vi+vj) - (ui+uj)*rhatij)/2.0;

  Pstar = rhoj*cj * (uj - ustar) + Pj;

  // that'll do it hopefully my dumb wave speed isn't too dumb
  }

}
