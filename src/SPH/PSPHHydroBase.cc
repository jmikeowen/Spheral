//---------------------------------Spheral++----------------------------------//
// PSPHHydroBase -- The PSPH/APSPH hydrodynamic package for Spheral++.
//
// Created by JMO, Wed Dec 16 20:52:02 PST 2015
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "computeSPHSumMassDensity.hh"
#include "computeSumVoronoiCellMassDensity.hh"
#include "computePSPHCorrections.hh"
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
#include "Hydro/PositionPolicy.hh"
#include "Hydro/PressurePolicy.hh"
#include "Hydro/SoundSpeedPolicy.hh"
#include "Hydro/GammaPolicy.hh"
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
#include "Mesh/Mesh.hh"
#include "CRKSPH/volumeSpacing.hh"

#include "PSPHHydroBase.hh"

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

namespace Spheral {

//------------------------------------------------------------------------------
// Construct with the given artificial viscosity and kernels.
//------------------------------------------------------------------------------
template<typename Dimension>
PSPHHydroBase<Dimension>::
PSPHHydroBase(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
              DataBase<Dimension>& dataBase,
              ArtificialViscosity<Dimension>& Q,
              const TableKernel<Dimension>& W,
              const TableKernel<Dimension>& WPi,
              const double filter,
              const double cfl,
              const bool useVelocityMagnitudeForDt,
              const bool compatibleEnergyEvolution,
              const bool evolveTotalEnergy,
              const bool XSPH,
              const bool correctVelocityGradient,
              const bool HopkinsConductivity,
              const bool sumMassDensityOverAllNodeLists,
              const MassDensityType densityUpdate,
              const HEvolutionType HUpdate,
              const Vector& xmin,
              const Vector& xmax):
  SPHHydroBase<Dimension>(smoothingScaleMethod,
                          dataBase,
                          Q,
                          W,
                          WPi,
                          filter,
                          cfl,
                          useVelocityMagnitudeForDt,
                          compatibleEnergyEvolution,
                          evolveTotalEnergy,
                          true,
                          XSPH,
                          correctVelocityGradient,
                          sumMassDensityOverAllNodeLists,
                          densityUpdate,
                          HUpdate,
                          0.0,
                          1.0,
                          xmin,
                          xmax),
  mHopkinsConductivity(HopkinsConductivity),
  mGamma(FieldStorageType::CopyFields),
  mPSPHcorrection(FieldStorageType::CopyFields) {

  // Create storage for our internal state.
  mGamma = dataBase.newFluidFieldList(0.0, HydroFieldNames::gamma);
  mPSPHcorrection = dataBase.newFluidFieldList(0.0, HydroFieldNames::PSPHcorrection);
  dataBase.fluidGamma(mGamma);
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
PSPHHydroBase<Dimension>::
~PSPHHydroBase() {
}

//------------------------------------------------------------------------------
// Register the state we need/are going to evolve.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
registerState(DataBase<Dimension>& dataBase,
              State<Dimension>& state) {

  typedef typename State<Dimension>::PolicyPointer PolicyPointer;

  // Make sure we're the right size.
  dataBase.resizeFluidFieldList(mGamma, 0.0, HydroFieldNames::gamma, false);
  dataBase.resizeFluidFieldList(mPSPHcorrection, 0.0, HydroFieldNames::PSPHcorrection, false);

  // SPH does most of it.
  SPHHydroBase<Dimension>::registerState(dataBase, state);

  // We also require the fluid gamma.
  PolicyPointer gammaPolicy(new GammaPolicy<Dimension>());
  state.enroll(mGamma, gammaPolicy);

  // Override the default policies for pressure and sound speed.  We'll compute those
  // specially in the postStateUpdate.  Same goes for registering the PSPHcorrections.
  state.enroll(mPSPHcorrection);
  state.removePolicy(this->mPressure);
  state.removePolicy(this->mSoundSpeed);
}

//------------------------------------------------------------------------------
// Pre-step initializations.  Since the topology has just been changed we need
// to recompute the PSPH corrections.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
preStepInitialize(const DataBase<Dimension>& dataBase,
                  State<Dimension>& state,
                  StateDerivatives<Dimension>& derivs) {

  // Base class
  SPHHydroBase<Dimension>::preStepInitialize(dataBase, state, derivs);

  // Do the PSPH corrections.
  const TableKernel<Dimension>& W = this->kernel();
  const ConnectivityMap<Dimension>& connectivityMap = dataBase.connectivityMap();
  const FieldList<Dimension, Scalar> mass = state.fields(HydroFieldNames::mass, 0.0);
  const FieldList<Dimension, Vector> position = state.fields(HydroFieldNames::position, Vector::zero);
  const FieldList<Dimension, Scalar> specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  const FieldList<Dimension, Scalar> gamma = state.fields(HydroFieldNames::gamma, 0.0);
  const FieldList<Dimension, SymTensor> H = state.fields(HydroFieldNames::H, SymTensor::zero);
  FieldList<Dimension, Scalar> rho = state.fields(HydroFieldNames::massDensity, 0.0);
  FieldList<Dimension, Scalar> P = state.fields(HydroFieldNames::pressure, 0.0);
  FieldList<Dimension, Scalar> cs = state.fields(HydroFieldNames::soundSpeed, 0.0);
  FieldList<Dimension, Scalar> PSPHcorrection = state.fields(HydroFieldNames::PSPHcorrection, 0.0);
  computePSPHCorrections(connectivityMap, W, mass, position, specificThermalEnergy, gamma, H, 
                         (this->mDensityUpdate != MassDensityType::IntegrateDensity),
                         rho, P, cs, PSPHcorrection);
  for (ConstBoundaryIterator boundItr = this->boundaryBegin();
       boundItr != this->boundaryEnd();
       ++boundItr) {
    (*boundItr)->applyFieldListGhostBoundary(rho);
    (*boundItr)->applyFieldListGhostBoundary(P);
    (*boundItr)->applyFieldListGhostBoundary(cs);
    (*boundItr)->applyFieldListGhostBoundary(PSPHcorrection);
  }
  for (ConstBoundaryIterator boundItr = this->boundaryBegin();
       boundItr != this->boundaryEnd();
       ++boundItr) (*boundItr)->finalizeGhostBoundary();
}

//------------------------------------------------------------------------------
// Post-state update.  This is where we can recompute the PSPH pressure and
// corrections.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
postStateUpdate(const Scalar /*time*/, 
                const Scalar /*dt*/,
                const DataBase<Dimension>& dataBase, 
                State<Dimension>& state,
                StateDerivatives<Dimension>& /*derivatives*/) {

  // First we need out boundary conditions completed, which the time integrator hasn't 
  // verified yet.
  for (ConstBoundaryIterator boundItr = this->boundaryBegin();
       boundItr != this->boundaryEnd();
       ++boundItr) (*boundItr)->finalizeGhostBoundary();
  
  // Do the PSPH corrections.
  const TableKernel<Dimension>& W = this->kernel();
  const ConnectivityMap<Dimension>& connectivityMap = dataBase.connectivityMap();
  const FieldList<Dimension, Scalar> mass = state.fields(HydroFieldNames::mass, 0.0);
  const FieldList<Dimension, Vector> position = state.fields(HydroFieldNames::position, Vector::zero);
  const FieldList<Dimension, Scalar> specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  const FieldList<Dimension, Scalar> gamma = state.fields(HydroFieldNames::gamma, 0.0);
  const FieldList<Dimension, SymTensor> H = state.fields(HydroFieldNames::H, SymTensor::zero);
  FieldList<Dimension, Scalar> rho = state.fields(HydroFieldNames::massDensity, 0.0);
  FieldList<Dimension, Scalar> P = state.fields(HydroFieldNames::pressure, 0.0);
  FieldList<Dimension, Scalar> cs = state.fields(HydroFieldNames::soundSpeed, 0.0);
  FieldList<Dimension, Scalar> PSPHcorrection = state.fields(HydroFieldNames::PSPHcorrection, 0.0);
  computePSPHCorrections(connectivityMap, W, mass, position, specificThermalEnergy, gamma, H,
                         (this->mDensityUpdate != MassDensityType::IntegrateDensity),
                         rho, P, cs, PSPHcorrection);
  for (ConstBoundaryIterator boundItr = this->boundaryBegin();
       boundItr != this->boundaryEnd();
       ++boundItr) {
    (*boundItr)->applyFieldListGhostBoundary(rho);
    (*boundItr)->applyFieldListGhostBoundary(P);
    (*boundItr)->applyFieldListGhostBoundary(cs);
    (*boundItr)->applyFieldListGhostBoundary(PSPHcorrection);
  }

  // We depend on the caller knowing to finalize the ghost boundaries!
}

//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
evaluateDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar /*dt*/,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivatives) const {

  // Get the ArtificialViscosity.
  auto& Q = this->artificialViscosity();

  // The kernels and such.
  const auto& W = this->kernel();
  const auto& WQ = this->PiKernel();

  // A few useful constants we'll use in the following loop.
  const double tiny = 1.0e-10;
  const Scalar W0 = W(0.0, 1.0);
  const auto compatibleEnergy = this->compatibleEnergyEvolution();
  const auto XSPH = this->XSPH();

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
  const auto specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  const auto H = state.fields(HydroFieldNames::H, SymTensor::zero);
  const auto pressure = state.fields(HydroFieldNames::pressure, 0.0);
  const auto soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);
  const auto gamma = state.fields(HydroFieldNames::gamma, 0.0);
  const auto PSPHcorrection = state.fields(HydroFieldNames::PSPHcorrection, 0.0);
  const auto reducingViscosityMultiplierQ = state.fields(HydroFieldNames::ArtificialViscousCqMultiplier, 0.0);
  const auto reducingViscosityMultiplierL = state.fields(HydroFieldNames::ArtificialViscousClMultiplier, 0.0);

  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(velocity.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(specificThermalEnergy.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);
  CHECK(pressure.size() == numNodeLists);
  CHECK(soundSpeed.size() == numNodeLists);
  CHECK(gamma.size() == numNodeLists);
  CHECK(PSPHcorrection.size() == numNodeLists);
  CHECK((not mHopkinsConductivity) or (reducingViscosityMultiplierQ.size() == numNodeLists));
  CHECK((not mHopkinsConductivity) or (reducingViscosityMultiplierL.size() == numNodeLists));

  // Derivative FieldLists.
  auto  rhoSum = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto  normalization = derivatives.fields(HydroFieldNames::normalization, 0.0);
  auto  DxDt = derivatives.fields(IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, Vector::zero);
  auto  DrhoDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto  DvDt = derivatives.fields(HydroFieldNames::hydroAcceleration, Vector::zero);
  auto  DepsDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy, 0.0);
  auto  DvDx = derivatives.fields(HydroFieldNames::velocityGradient, Tensor::zero);
  auto  localDvDx = derivatives.fields(HydroFieldNames::internalVelocityGradient, Tensor::zero);
  auto  M = derivatives.fields(HydroFieldNames::M_SPHCorrection, Tensor::zero);
  auto  localM = derivatives.fields("local " + HydroFieldNames::M_SPHCorrection, Tensor::zero);
  auto  DHDt = derivatives.fields(IncrementFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto  Hideal = derivatives.fields(ReplaceBoundedFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto  maxViscousPressure = derivatives.fields(HydroFieldNames::maxViscousPressure, 0.0);
  auto  effViscousPressure = derivatives.fields(HydroFieldNames::effectiveViscousPressure, 0.0);
  auto  viscousWork = derivatives.fields(HydroFieldNames::viscousWork, 0.0);
  auto& pairAccelerations = derivatives.getAny(HydroFieldNames::pairAccelerations, vector<Vector>());
  auto  XSPHWeightSum = derivatives.fields(HydroFieldNames::XSPHWeightSum, 0.0);
  auto  XSPHDeltaV = derivatives.fields(HydroFieldNames::XSPHDeltaV, Vector::zero);
  auto  weightedNeighborSum = derivatives.fields(HydroFieldNames::weightedNeighborSum, 0.0);
  auto  massSecondMoment = derivatives.fields(HydroFieldNames::massSecondMoment, SymTensor::zero);
  CHECK(rhoSum.size() == numNodeLists);
  CHECK(normalization.size() == numNodeLists);
  CHECK(DxDt.size() == numNodeLists);
  CHECK(DrhoDt.size() == numNodeLists);
  CHECK(DvDt.size() == numNodeLists);
  CHECK(DepsDt.size() == numNodeLists);
  CHECK(DvDx.size() == numNodeLists);
  CHECK(localDvDx.size() == numNodeLists);
  CHECK(M.size() == numNodeLists);
  CHECK(localM.size() == numNodeLists);
  CHECK(DHDt.size() == numNodeLists);
  CHECK(Hideal.size() == numNodeLists);
  CHECK(maxViscousPressure.size() == numNodeLists);
  CHECK(effViscousPressure.size() == numNodeLists);
  CHECK(viscousWork.size() == numNodeLists);
  CHECK(XSPHWeightSum.size() == numNodeLists);
  CHECK(XSPHDeltaV.size() == numNodeLists);
  CHECK(weightedNeighborSum.size() == numNodeLists);
  CHECK(massSecondMoment.size() == numNodeLists);

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();

  // Size up the pair-wise accelerations before we start.
  if (compatibleEnergy) pairAccelerations = vector<Vector>(npairs);

  // Walk all the interacting pairs.
#pragma omp parallel
  {
    // Thread private scratch variables
    int i, j, nodeListi, nodeListj;
    Scalar Wi, gWi, WQi, gWQi, Wj, gWj, WQj, gWQj;
    Tensor QPiij, QPiji;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;
    auto rhoSum_thread = rhoSum.threadCopy(threadStack);
    auto normalization_thread = normalization.threadCopy(threadStack);
    auto DvDt_thread = DvDt.threadCopy(threadStack);
    auto DepsDt_thread = DepsDt.threadCopy(threadStack);
    auto DvDx_thread = DvDx.threadCopy(threadStack);
    auto localDvDx_thread = localDvDx.threadCopy(threadStack);
    auto M_thread = M.threadCopy(threadStack);
    auto localM_thread = localM.threadCopy(threadStack);
    auto maxViscousPressure_thread = maxViscousPressure.threadCopy(threadStack, ThreadReduction::MAX);
    auto effViscousPressure_thread = effViscousPressure.threadCopy(threadStack);
    auto viscousWork_thread = viscousWork.threadCopy(threadStack);
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
      const auto& epsi = specificThermalEnergy(nodeListi, i);
      const auto& Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto& ci = soundSpeed(nodeListi, i);
      const auto& gammai = gamma(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& rhoSumi = rhoSum_thread(nodeListi, i);
      auto& normi = normalization_thread(nodeListi, i);
      auto& DvDti = DvDt_thread(nodeListi, i);
      auto& DepsDti = DepsDt_thread(nodeListi, i);
      auto& DvDxi = DvDx_thread(nodeListi, i);
      auto& localDvDxi = localDvDx_thread(nodeListi, i);
      auto& Mi = M_thread(nodeListi, i);
      auto& localMi = localM_thread(nodeListi, i);
      auto& maxViscousPressurei = maxViscousPressure_thread(nodeListi, i);
      auto& effViscousPressurei = effViscousPressure_thread(nodeListi, i);
      auto& viscousWorki = viscousWork_thread(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum_thread(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV_thread(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum_thread(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment_thread(nodeListi, i);

      // Get the state for node j
      const auto& rj = position(nodeListj, j);
      const auto& mj = mass(nodeListj, j);
      const auto& vj = velocity(nodeListj, j);
      const auto& rhoj = massDensity(nodeListj, j);
      const auto& epsj = specificThermalEnergy(nodeListj, j);
      const auto& Pj = pressure(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto& cj = soundSpeed(nodeListj, j);
      const auto& gammaj = gamma(nodeListj, j);
      const auto  Hdetj = Hj.Determinant();
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& rhoSumj = rhoSum_thread(nodeListj, j);
      auto& normj = normalization_thread(nodeListj, j);
      auto& DvDtj = DvDt_thread(nodeListj, j);
      auto& DepsDtj = DepsDt_thread(nodeListj, j);
      auto& DvDxj = DvDx_thread(nodeListj, j);
      auto& localDvDxj = localDvDx_thread(nodeListj, j);
      auto& Mj = M_thread(nodeListj, j);
      auto& localMj = localM_thread(nodeListj, j);
      auto& maxViscousPressurej = maxViscousPressure_thread(nodeListj, j);
      auto& effViscousPressurej = effViscousPressure_thread(nodeListj, j);
      auto& viscousWorkj = viscousWork_thread(nodeListj, j);
      auto& XSPHWeightSumj = XSPHWeightSum_thread(nodeListj, j);
      auto& XSPHDeltaVj = XSPHDeltaV_thread(nodeListj, j);
      auto& weightedNeighborSumj = weightedNeighborSum_thread(nodeListj, j);
      auto& massSecondMomentj = massSecondMoment_thread(nodeListj, j);

      // Flag if this is a contiguous material pair or not.
      const bool sameMatij = true; // (nodeListi == nodeListj and fragIDi == fragIDj);

      // Node displacement.
      const auto rij = ri - rj;
      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);

      // Symmetrized kernel weight and gradient.
      std::tie(Wi, gWi) = W.kernelAndGradValue(etaMagi, Hdeti);
      std::tie(WQi, gWQi) = WQ.kernelAndGradValue(etaMagi, Hdeti);
      const auto Hetai = Hi*etai.unitVector();
      const auto gradWi = gWi*Hetai;
      const auto gradWQi = gWQi*Hetai;

      std::tie(Wj, gWj) = W.kernelAndGradValue(etaMagj, Hdetj);
      std::tie(WQj, gWQj) = WQ.kernelAndGradValue(etaMagj, Hdetj);
      const auto Hetaj = Hj*etaj.unitVector();
      const auto gradWj = gWj*Hetaj;
      const auto gradWQj = gWQj*Hetaj;

      // Zero'th and second moment of the node distribution -- used for the
      // ideal H calculation.
      const auto fweightij = sameMatij ? 1.0 : mj*rhoi/(mi*rhoj);
      const auto rij2 = rij.magnitude2();
      const auto thpt = rij.selfdyad()*safeInvVar(rij2*rij2*rij2);
      weightedNeighborSumi +=     fweightij*std::abs(gWi);
      weightedNeighborSumj += 1.0/fweightij*std::abs(gWj);
      massSecondMomenti +=     fweightij*gradWi.magnitude2()*thpt;
      massSecondMomentj += 1.0/fweightij*gradWj.magnitude2()*thpt;


      // Contribution to the sum density.
      if (nodeListi == nodeListj) {
        rhoSumi += mj*Wj;
        rhoSumj += mi*Wi;
        normi += mi/rhoi*Wj;
        normj += mj/rhoj*Wi;
      }

      // Compute the pair-wise artificial viscosity.
      const auto vij = vi - vj;
      std::tie(QPiij, QPiji) = Q.Piij(nodeListi, i, nodeListj, j,
                                      ri, etai, vi, rhoi, ci, Hi,
                                      rj, etaj, vj, rhoj, cj, Hj);
      const auto Qacci = 0.5*(QPiij*gradWQi);
      const auto Qaccj = 0.5*(QPiji*gradWQj);
      // const auto workQi = 0.5*(QPiij*vij).dot(gradWQi);
      // const auto workQj = 0.5*(QPiji*vij).dot(gradWQj);
      const auto workQi = vij.dot(Qacci);
      const auto workQj = vij.dot(Qaccj);
      const auto Qi = rhoi*rhoi*(QPiij.diagonalElements().maxAbsElement());
      const auto Qj = rhoj*rhoj*(QPiji.diagonalElements().maxAbsElement());
      maxViscousPressurei = max(maxViscousPressurei, Qi);
      maxViscousPressurej = max(maxViscousPressurej, Qj);
      effViscousPressurei += mj*Qi*WQi/rhoj;
      effViscousPressurej += mi*Qj*WQj/rhoi;
      viscousWorki += mj*workQi;
      viscousWorkj += mi*workQj;

      // Acceleration.
      CHECK(rhoi > 0.0);
      CHECK(rhoj > 0.0);
      const auto& Fcorri=PSPHcorrection(nodeListi, i);
      const auto& Fcorrj=PSPHcorrection(nodeListj, j);
      const auto  Fij=1.0-Fcorri*safeInv(mj*epsj, tiny);
      const auto  Fji=1.0-Fcorrj*safeInv(mi*epsi, tiny);
      const auto  engCoef=(gammai-1)*(gammaj-1)*epsi*epsj;
      const auto  deltaDvDt = engCoef*(gradWi*Fij*safeInv(Pi, tiny) + gradWj*Fji*safeInv(Pj, tiny)) + Qacci + Qaccj;
      DvDti -= mj*deltaDvDt;
      DvDtj += mi*deltaDvDt;
      if (compatibleEnergy) pairAccelerations[kk] = -mj*deltaDvDt;  // Acceleration for i (j anti-symmetric)

      // Specific thermal energy evolution.
      DepsDti += mj*(engCoef*Fij*safeInv(Pi, tiny)*vij.dot(gradWi) + workQi);
      DepsDtj += mi*(engCoef*Fji*safeInv(Pj, tiny)*vij.dot(gradWj) + workQj);

      //ADD ARITIFICIAL CONDUCTIVITY IN HOPKINS 2014A
      if (mHopkinsConductivity) {
        const auto  alph_c = 0.25;//Parameter = 0.25 in Hopkins 2014
        const auto  Vs = ci+cj-3.0*vij.dot(rij.unitVector());
        const auto& Qalpha_i = reducingViscosityMultiplierL(nodeListi, i); //Both L and Q corrections are the same for Cullen Viscosity
        const auto& Qalpha_j = reducingViscosityMultiplierL(nodeListj, j); //Both L and Q corrections are the same for Cullen Viscosity
        //DepsDti += (Vs > 0.0)*alph_c*mi*mj*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsi-epsj)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj+1e-30)*(rhoi+rhoj+1e-30));
        //DepsDtj += (Vs > 0.0)*alph_c*mi*mj*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsi-epsj)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj+1e-30)*(rhoi+rhoj+1e-30));
        //DepsDti += (Vs > 0.0) ? alph_c*mj*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsi-epsj)*abs(Pi-Pj)*((gradWi).dot(rij.unitVector()))/max((Pi+Pj)*0.5*(rhoi+rhoj),tiny) : 0.0;
        //DepsDtj += (Vs > 0.0) ? alph_c*mi*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsj-epsi)*abs(Pi-Pj)*((gradWj).dot(rij.unitVector()))/max((Pi+Pj)*0.5*(rhoi+rhoj),tiny) : 0.0;
        DepsDti += (Vs > 0.0) ? alph_c*mj*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsi-epsj)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj)*(rhoi+rhoj),tiny) : 0.0;
        DepsDtj += (Vs > 0.0) ? alph_c*mi*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsj-epsi)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj)*(rhoi+rhoj),tiny) : 0.0;
        //const Scalar tmpi = (Vs > 0.0) ? alph_c*mj*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsi-epsj)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj)*(rhoi+rhoj)) : 0.0;
        //const Scalar tmpj = (Vs > 0.0) ? alph_c*mi*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsj-epsi)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj)*(rhoi+rhoj)) : 0.0;
        //DepsDti += tmpi;
        //DepsDtj += tmpj;
        //DepsDti += (Vs > 0.0) ? alph_c*mj*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsi-epsj)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj)*(rhoi+rhoj)) : 0.0;
        //DepsDtj += (Vs > 0.0) ? alph_c*mi*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsj-epsi)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj)*(rhoi+rhoj)) : 0.0;
        //DepsDti += (Vs > 0.0 && Pi > 1e-4 && Pj > 1e-4) ? alph_c*mj*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsi-epsj)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj)*(rhoi+rhoj),tiny) : 0.0;
        //DepsDtj += (Vs > 0.0 && Pi > 1e-4 && Pj > 1e-4) ? alph_c*mi*(Qalpha_i+Qalpha_j)*0.5*Vs*(epsj-epsi)*abs(Pi-Pj)*((gradWi+gradWj).dot(rij.unitVector()))*safeInv((Pi+Pj)*(rhoi+rhoj),tiny) : 0.0;
      }

      // Velocity gradient.
      const auto deltaDvDxi = mj*vij.dyad(gradWi);
      const auto deltaDvDxj = mi*vij.dyad(gradWj);
      DvDxi -= deltaDvDxi; 
      DvDxj -= deltaDvDxj;
      if (sameMatij) {
        localDvDxi -= deltaDvDxi; 
        localDvDxj -= deltaDvDxj;
      }

      // Estimate of delta v (for XSPH).
      if (XSPH and (sameMatij)) {
        const auto wXSPHij = 0.5*(mi/rhoi*Wi + mj/rhoj*Wj);
        XSPHWeightSumi += wXSPHij;
        XSPHWeightSumj += wXSPHij;
        XSPHDeltaVi -= wXSPHij*vij;
        XSPHDeltaVj += wXSPHij*vij;
      }

      // Linear gradient correction term.
      Mi -= mj*rij.dyad(gradWi);
      Mj -= mi*rij.dyad(gradWj);
      if (sameMatij) {
        localMi -= mj*rij.dyad(gradWi);
        localMj -= mi*rij.dyad(gradWj);
      }

    } // loop over pairs

    // Reduce the thread values to the master.
    threadReduceFieldLists<Dimension>(threadStack);

  }   // OpenMP parallel region

  // Finish up the derivatives for each point.
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
      const auto numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& rhoSumi = rhoSum(nodeListi, i);
      auto& normi = normalization(nodeListi, i);
      auto& DxDti = DxDt(nodeListi, i);
      auto& DrhoDti = DrhoDt(nodeListi, i);
      auto& DvDti = DvDt(nodeListi, i);
      auto& DepsDti = DepsDt(nodeListi, i);
      auto& DvDxi = DvDx(nodeListi, i);
      auto& localDvDxi = localDvDx(nodeListi, i);
      auto& Mi = M(nodeListi, i);
      auto& localMi = localM(nodeListi, i);
      auto& DHDti = DHDt(nodeListi, i);
      auto& Hideali = Hideal(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment(nodeListi, i);

      // Add the self-contribution to density sum.
      rhoSumi += mi*W0*Hdeti;
      normi += mi/rhoi*W0*Hdeti;

      // Finish the gradient of the velocity.
      CHECK(rhoi > 0.0);
      if (this->mCorrectVelocityGradient and
          std::abs(Mi.Determinant()) > 1.0e-10 and
          numNeighborsi > Dimension::pownu(2)) {
        Mi = Mi.Inverse();
        DvDxi = DvDxi*Mi;
      } else {
        DvDxi /= rhoi;
      }
      if (this->mCorrectVelocityGradient and
          std::abs(localMi.Determinant()) > 1.0e-10 and
          numNeighborsi > Dimension::pownu(2)) {
        localMi = localMi.Inverse();
        localDvDxi = localDvDxi*localMi;
      } else {
        localDvDxi /= rhoi;
      }

      // Evaluate the continuity equation.
      DrhoDti = -rhoi*DvDxi.Trace();

      // If needed finish the total energy derivative.
      if (this->mEvolveTotalEnergy) DepsDti = mi*(vi.dot(DvDti) + DepsDti);

      // Complete the moments of the node distribution for use in the ideal H calculation.
      weightedNeighborSumi = Dimension::rootnu(max(0.0, weightedNeighborSumi/Hdeti));
      massSecondMomenti /= Hdeti*Hdeti;

      // Determine the position evolution, based on whether we're doing XSPH or not.
      if (this->XSPH()) {
        XSPHWeightSumi += Hdeti*mi/rhoi*W0;
        CHECK2(XSPHWeightSumi != 0.0, i << " " << XSPHWeightSumi);
        DxDti = vi + XSPHDeltaVi*safeInvVar(XSPHWeightSumi, tiny);
      } else {
        DxDti = vi;
      }

      // The H tensor evolution.
      DHDti = this->mSmoothingScaleMethod.smoothingScaleDerivative(Hi,
                                                                   ri,
                                                                   DvDxi,
                                                                   hmin,
                                                                   hmax,
                                                                   hminratio,
                                                                   nPerh);
      Hideali = this->mSmoothingScaleMethod.newSmoothingScale(Hi,
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
}

//------------------------------------------------------------------------------
// Finalize the derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
finalizeDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar /*dt*/,
                    const DataBase<Dimension>& /*dataBase*/,
                    const State<Dimension>& /*state*/,
                    StateDerivatives<Dimension>& derivs) const {

  // If we're using the compatible energy discretization, we need to enforce
  // boundary conditions on the accelerations.
  if (this->mCompatibleEnergyEvolution) {
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
}

//------------------------------------------------------------------------------
// Apply the ghost boundary conditions for hydro state fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
applyGhostBoundaries(State<Dimension>& state,
                     StateDerivatives<Dimension>& derivs) {

  // SPH does most of it.
  SPHHydroBase<Dimension>::applyGhostBoundaries(state, derivs);

  // Apply boundary conditions to the PSPH corrections.
  FieldList<Dimension, Scalar> gamma = state.fields(HydroFieldNames::gamma, 0.0);
  FieldList<Dimension, Scalar> PSPHcorrection = state.fields(HydroFieldNames::PSPHcorrection, 0.0);

  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
    (*boundaryItr)->applyFieldListGhostBoundary(gamma);
    (*boundaryItr)->applyFieldListGhostBoundary(PSPHcorrection);
  }
}

//------------------------------------------------------------------------------
// Enforce the boundary conditions for hydro state fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
enforceBoundaries(State<Dimension>& state,
                  StateDerivatives<Dimension>& derivs) {

  // SPH does most of it.
  SPHHydroBase<Dimension>::enforceBoundaries(state, derivs);

  // Enforce boundary conditions on the PSPH corrections.
  FieldList<Dimension, Scalar> gamma = state.fields(HydroFieldNames::gamma, 0.0);
  FieldList<Dimension, Scalar> PSPHcorrection = state.fields(HydroFieldNames::PSPHcorrection, 0.0);

  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
    (*boundaryItr)->enforceFieldListBoundary(gamma);
    (*boundaryItr)->enforceFieldListBoundary(PSPHcorrection);
  }
}

//------------------------------------------------------------------------------
// Dump the current state to the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
dumpState(FileIO& file, const string& pathName) const {

  // SPH does most of it.
  SPHHydroBase<Dimension>::dumpState(file, pathName);

  file.write(mGamma, pathName + "/gamma");
  file.write(mPSPHcorrection, pathName + "/PSPHcorrection");
}

//------------------------------------------------------------------------------
// Restore the state from the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
PSPHHydroBase<Dimension>::
restoreState(const FileIO& file, const string& pathName) {
 
  // SPH does most of it.
  SPHHydroBase<Dimension>::restoreState(file, pathName);

  file.read(mGamma, pathName + "/gamma");
  file.read(mPSPHcorrection, pathName + "/PSPHcorrection");
}

}
