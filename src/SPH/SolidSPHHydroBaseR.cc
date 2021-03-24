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
#include "FileIO/FileIO.hh"
#include "NodeList/SmoothingScaleBase.hh"
#include "Hydro/HydroFieldNames.hh"
// #include "Hydro/RNonSymmetricSpecificThermalEnergyPolicy.hh"
#include "Strength/SolidFieldNames.hh"
#include "NodeList/SolidNodeList.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/IncrementFieldList.hh"
#include "DataBase/IncrementBoundedFieldList.hh"
#include "DataBase/ReplaceFieldList.hh"
#include "DataBase/ReplaceBoundedFieldList.hh"
#include "ArtificialViscosity/ArtificialViscosity.hh"
#include "DataBase/DataBase.hh"
#include "Field/FieldList.hh"
#include "Field/NodeIterators.hh"
#include "Boundary/Boundary.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Utilities/timingUtilities.hh"
#include "Utilities/safeInv.hh"
#include "SolidMaterial/SolidEquationOfState.hh"
#include "Geometry/GeometryRegistrar.hh"

#include "SolidSPHHydroBaseR.hh"

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
// Compute the artificial tensile stress correction tensor for the given 
// stress tensor
//------------------------------------------------------------------------------
inline
Dim<1>::SymTensor
tensileStressCorrection(const Dim<1>::SymTensor& sigma) {
  const auto lambdax = sigma.xx();
  return Dim<1>::SymTensor(lambdax > 0.0 ? -lambdax : 0.0);
}

//------------------------------------------------------------------------------
// Construct with the given artificial viscosity and kernels.
//------------------------------------------------------------------------------
SolidSPHHydroBaseR::
SolidSPHHydroBaseR(const SmoothingScaleBase<Dim<1>>& smoothingScaleMethod,
                   DataBase<Dimension>& dataBase,
                   ArtificialViscosity<Dim<1>>& Q,
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
                   const Vector& xmax):
  SolidSPHHydroBase<Dim<1>>(smoothingScaleMethod, 
                            dataBase,
                            Q,
                            W.kernel(),
                            WPi.kernel(),
                            WGrad.kernel(),
                            filter,
                            cfl,
                            useVelocityMagnitudeForDt,
                            compatibleEnergyEvolution,
                            evolveTotalEnergy,
                            gradhCorrection,
                            XSPH,
                            correctVelocityGradient,
                            sumMassDensityOverAllNodeLists,
                            densityUpdate,
                            HUpdate,
                            epsTensile,
                            nTensile,
                            damageRelieveRubble,
                            negativePressureInDamage,
                            strengthInDamage,
                            xmin,
                            xmax) {
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
SolidSPHHydroBaseR::
~SolidSPHHydroBaseR() {
}

//------------------------------------------------------------------------------
// On problem start up, we need to initialize our internal data.
//------------------------------------------------------------------------------
void
SolidSPHHydroBaseR::
initializeProblemStartup(DataBase<Dim<1>>& dataBase) {
  GeometryRegistrar::coords(CoordinateType::Spherical);
  SolidSPHHydroBase<Dim<1>>::initializeProblemStartup(dataBase);
}

//------------------------------------------------------------------------------
// Register the state we need/are going to evolve.
//------------------------------------------------------------------------------
void
SolidSPHHydroBaseR::
registerState(DataBase<Dim<1>>& dataBase,
              State<Dim<1>>& state) {

  // typedef State<Dimension>::PolicyPointer PolicyPointer;

  // Call the ancestor.
  SolidSPHHydroBase<Dim<1>>::registerState(dataBase, state);

  // // Reregister the plastic strain policy to the R specialized version
  // // that accounts for the non-R components of the stress.  Also reregister
  // // the deviatoric stress to not zero out the trace.
  // auto S = state.fields(SolidFieldNames::deviatoricStress, SymTensor::zero);
  // PolicyPointer plasticStrainPolicy(new SphericalPlasticStrainPolicy());
  // PolicyPointer deviatoricStressPolicy(new DeviatoricStressPolicy<Dim<1>>(false));
  // state.enroll(ps, plasticStrainPolicy);
  // state.enroll(S, deviatoricStressPolicy);

  // Are we using the compatible energy evolution scheme?
  // If so we need to override the ordinary energy registration with a specialized version.
  // if (mCompatibleEnergyEvolution) {
  //   FieldList<Dimension, Scalar> specificThermalEnergy = dataBase.fluidSpecificThermalEnergy();
  //   PolicyPointer thermalEnergyPolicy(new RNonSymmetricSpecificThermalEnergyPolicy(dataBase));
  //   state.enroll(specificThermalEnergy, thermalEnergyPolicy);

  //   // Get the policy for the position, and add the specific energy as a dependency.
  //   PolicyPointer positionPolicy = state.policy(state.buildFieldKey(HydroFieldNames::position, UpdatePolicyBase<Dimension>::wildcard()));
  //   positionPolicy->addDependency(HydroFieldNames::specificThermalEnergy);
  // }
}

//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
void
SolidSPHHydroBaseR::
evaluateDerivatives(const Dim<1>::Scalar /*time*/,
                    const Dim<1>::Scalar dt,
                    const DataBase<Dim<1>>& dataBase,
                    const State<Dim<1>>& state,
                    StateDerivatives<Dim<1>>& derivatives) const {

  // Get the ArtificialViscosity.
  auto& Q = this->artificialViscosity();

  // The kernels and such.
  const auto& W = *mKernelPtr;
  const auto& WQ = *mPiKernelPtr;
  const auto& WG = *mGKernelPtr;
  const auto& smoothingScaleMethod = this->smoothingScaleMethod();
  const auto  uniqueWQ = (mPiKernelPtr != mKernelPtr);
  const auto  uniqueWG = (mGKernelPtr != mKernelPtr);

  // A few useful constants we'll use in the following loop.
  const auto tiny = 1.0e-30;
  const auto epsTensile = this->epsilonTensile();
  const auto compatibleEnergy = this->compatibleEnergyEvolution();
  const auto XSPH = this->XSPH();
  const auto damageRelieveRubble = this->damageRelieveRubble();

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
  const auto omega = state.fields(HydroFieldNames::omegaGradh, 0.0);
  const auto S = state.fields(SolidFieldNames::deviatoricStress, SymTensor::zero);
  const auto mu = state.fields(SolidFieldNames::shearModulus, 0.0);
  const auto damage = state.fields(SolidFieldNames::tensorDamage, SymTensor::zero);
  const auto fragIDs = state.fields(SolidFieldNames::fragmentIDs, int(1));
  const auto pTypes = state.fields(SolidFieldNames::particleTypes, int(0));
  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(velocity.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(specificThermalEnergy.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);
  CHECK(pressure.size() == numNodeLists);
  CHECK(soundSpeed.size() == numNodeLists);
  CHECK(omega.size() == numNodeLists);
  CHECK(S.size() == numNodeLists);
  CHECK(mu.size() == numNodeLists);
  CHECK(damage.size() == numNodeLists);
  CHECK(fragIDs.size() == numNodeLists);
  CHECK(pTypes.size() == numNodeLists);

  // const auto& Hfield0 = this->Hfield0();

  // Derivative FieldLists.
  auto  rhoSum = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
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
  auto  rhoSumCorrection = derivatives.fields(HydroFieldNames::massDensityCorrection, 0.0);
  auto  viscousWork = derivatives.fields(HydroFieldNames::viscousWork, 0.0);
  auto& pairAccelerations = derivatives.getAny(HydroFieldNames::pairAccelerations, vector<Vector>());
  auto  XSPHWeightSum = derivatives.fields(HydroFieldNames::XSPHWeightSum, 0.0);
  auto  XSPHDeltaV = derivatives.fields(HydroFieldNames::XSPHDeltaV, Vector::zero);
  auto  weightedNeighborSum = derivatives.fields(HydroFieldNames::weightedNeighborSum, 0.0);
  auto  massSecondMoment = derivatives.fields(HydroFieldNames::massSecondMoment, SymTensor::zero);
  auto  DSDt = derivatives.fields(IncrementFieldList<Dimension, SymTensor>::prefix() + SolidFieldNames::deviatoricStress, SymTensor::zero);
  CHECK(rhoSum.size() == numNodeLists);
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
  CHECK(rhoSumCorrection.size() == numNodeLists);
  CHECK(viscousWork.size() == numNodeLists);
  CHECK(XSPHWeightSum.size() == numNodeLists);
  CHECK(XSPHDeltaV.size() == numNodeLists);
  CHECK(weightedNeighborSum.size() == numNodeLists);
  CHECK(massSecondMoment.size() == numNodeLists);
  CHECK(DSDt.size() == numNodeLists);

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();
  // const auto& coupling = connectivityMap.coupling();

  // Size up the pair-wise accelerations before we start.
  if (compatibleEnergy) pairAccelerations.resize(2*npairs + dataBase.numInternalNodes()); // Also have to account for self-interaction

  // The scale for the tensile correction.
  const auto& nodeList = mass[0]->nodeList();
  const auto  nPerh = nodeList.nodesPerSmoothingScale();
  const auto  WnPerh = W.kernelValue(1.0/nPerh, 1.0);

  // Walk all the interacting pairs.
#pragma omp parallel
  {
    // Thread private  scratch variables.
    int i, j, nodeListi, nodeListj;
    Scalar Wi, Wj, WQi, WQj;
    Vector gradWi, gradWj, gradWQi, gradWQj, gradWGi, gradWGj;
    Tensor QPiij, QPiji;
    SymTensor sigmai, sigmaj;

    typename SpheralThreads<Dim<1>>::FieldListStack threadStack;
    auto rhoSum_thread = rhoSum.threadCopy(threadStack);
    auto DvDt_thread = DvDt.threadCopy(threadStack);
    auto DepsDt_thread = DepsDt.threadCopy(threadStack);
    auto DvDx_thread = DvDx.threadCopy(threadStack);
    auto localDvDx_thread = localDvDx.threadCopy(threadStack);
    auto M_thread = M.threadCopy(threadStack);
    auto localM_thread = localM.threadCopy(threadStack);
    auto maxViscousPressure_thread = maxViscousPressure.threadCopy(threadStack, ThreadReduction::MAX);
    auto effViscousPressure_thread = effViscousPressure.threadCopy(threadStack);
    auto rhoSumCorrection_thread = rhoSumCorrection.threadCopy(threadStack);
    auto viscousWork_thread = viscousWork.threadCopy(threadStack);
    auto XSPHWeightSum_thread = XSPHWeightSum.threadCopy(threadStack);
    auto XSPHDeltaV_thread = XSPHDeltaV.threadCopy(threadStack);
    auto weightedNeighborSum_thread = weightedNeighborSum.threadCopy(threadStack);
    auto massSecondMoment_thread = massSecondMoment.threadCopy(threadStack);
    auto DSDt_thread = DSDt.threadCopy(threadStack);

#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& posi = position(nodeListi, i);
      const auto  mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto  rhoi = massDensity(nodeListi, i);
      const auto  Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  ci = soundSpeed(nodeListi, i);
      const auto  omegai = omega(nodeListi, i);
      const auto& Si = S(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      const auto  safeOmegai = safeInv(omegai, tiny);
      //const auto  fragIDi = fragIDs(nodeListi, i);
      const auto  pTypei = pTypes(nodeListi, i);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& rhoSumi = rhoSum(nodeListi, i);
      auto& DvDti = DvDt(nodeListi, i);
      auto& DepsDti = DepsDt(nodeListi, i);
      auto& DvDxi = DvDx(nodeListi, i);
      auto& localDvDxi = localDvDx(nodeListi, i);
      auto& Mi = M(nodeListi, i);
      auto& localMi = localM(nodeListi, i);
      auto& maxViscousPressurei = maxViscousPressure(nodeListi, i);
      auto& effViscousPressurei = effViscousPressure(nodeListi, i);
      auto& rhoSumCorrectioni = rhoSumCorrection(nodeListi, i);
      auto& viscousWorki = viscousWork(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment(nodeListi, i);

      // Get the state for node j.
      const auto& posj = position(nodeListj, j);
      const auto  mj = mass(nodeListj, j);
      const auto& vj = velocity(nodeListj, j);
      const auto  rhoj = massDensity(nodeListj, j);
      const auto  Pj = pressure(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto  cj = soundSpeed(nodeListj, j);
      const auto  omegaj = omega(nodeListj, j);
      const auto& Sj = S(nodeListj, j);
      const auto  Hdetj = Hj.Determinant();
      const auto  safeOmegaj = safeInv(omegaj, tiny);
      //const auto  fragIDj = fragIDs(nodeListj, j);
      const auto  pTypej = pTypes(nodeListj, j);
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& rhoSumj = rhoSum(nodeListj, j);
      auto& DvDtj = DvDt(nodeListj, j);
      auto& DepsDtj = DepsDt(nodeListj, j);
      auto& DvDxj = DvDx(nodeListj, j);
      auto& localDvDxj = localDvDx(nodeListj, j);
      auto& Mj = M(nodeListj, j);
      auto& localMj = localM(nodeListj, j);
      auto& maxViscousPressurej = maxViscousPressure(nodeListj, j);
      auto& effViscousPressurej = effViscousPressure(nodeListj, j);
      auto& rhoSumCorrectionj = rhoSumCorrection(nodeListj, j);
      auto& viscousWorkj = viscousWork(nodeListj, j);
      auto& XSPHWeightSumj = XSPHWeightSum(nodeListj, j);
      auto& XSPHDeltaVj = XSPHDeltaV(nodeListj, j);
      auto& weightedNeighborSumj = weightedNeighborSum(nodeListj, j);
      auto& massSecondMomentj = massSecondMoment(nodeListj, j);

      // Flag if this is a contiguous material pair or not.
      const auto sameMatij = true; // (nodeListi == nodeListj and fragIDi == fragIDj);

      // Flag if at least one particle is free (0).
      const auto freeParticle = (pTypei == 0 or pTypej == 0);

      // Node displacement.
      const auto xij = posi - posj;

      // Symmetrized kernel weight and gradient.
      const auto etaiI = Hi*posi;
      const auto etajI = Hj*posj;
      const auto etaiJ = Hj*posi;
      const auto etajJ = Hj*posj;
      std::tie(Wi, gradWi) = W.kernelAndGradValue(etajI, etaiI, Hdeti);
      std::tie(Wj, gradWj) = W.kernelAndGradValue(etajJ, etaiJ, Hdetj);
      if (uniqueWQ) {
        std::tie(WQi, gradWQi) = W.kernelAndGradValue(etajI, etaiI, Hdeti);
        std::tie(WQj, gradWQj) = W.kernelAndGradValue(etajJ, etaiJ, Hdetj);
      } else {
        WQi = Wi;
        WQj = Wj;
        gradWQi = gradWi;
        gradWQj = gradWj;
      }
      gradWGi = uniqueWG ? WG.grad(etajI, etaiI, Hdeti) : gradWi;
      gradWGj = uniqueWG ? WG.grad(etajJ, etaiJ, Hdetj) : gradWj;

      // Determine how we're applying damage.
      const auto fDij = pairs[kk].f_couple;

      // Zero'th and second moment of the node distribution -- used for the
      // ideal H calculation.
      const auto fweightij = sameMatij ? 1.0 : mj*rhoi/(mi*rhoj);
      const auto xij2 = xij.magnitude2();
      const auto thpt = xij.selfdyad()*safeInvVar(xij2*xij2*xij2);
      weightedNeighborSumi +=     fweightij*abs(gradWi.x());
      weightedNeighborSumj += 1.0/fweightij*abs(gradWj.x());
      massSecondMomenti +=     fweightij*gradWi.magnitude2()*thpt;
      massSecondMomentj += 1.0/fweightij*gradWj.magnitude2()*thpt;

      // Contribution to the sum density (only if the same material).
      if (nodeListi == nodeListj) {
        rhoSumi += mj*Wi;
        rhoSumj += mi*Wj;
      }

      // Contribution to the sum density correction
      rhoSumCorrectioni += mj * WQi / rhoj ;
      rhoSumCorrectionj += mi * WQj / rhoi ;

      // Compute the pair-wise artificial viscosity.
      const auto vij = vi - vj;
      std::tie(QPiij, QPiji) = Q.Piij(nodeListi, i, nodeListj, j,
                                      posi, Hi*xij, vi, rhoi, ci, Hi,
                                      posj, Hj*xij, vj, rhoj, cj, Hj);
      const auto Qacci = 0.5*(QPiij*gradWQi);
      const auto Qaccj = 0.5*(QPiji*gradWQj);
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

      // Compute the stress tensors.
      if (sameMatij) {
        sigmai = fDij*Si - Pi * SymTensor::one;
        sigmaj = fDij*Sj - Pj * SymTensor::one;
      } else {
        sigmai = -Pi * SymTensor::one;
        sigmaj = -Pj * SymTensor::one;
      }

      // Compute the tensile correction to add to the stress as described in 
      // Gray, Monaghan, & Swift (Comput. Methods Appl. Mech. Eng., 190, 2001)
      const auto fi = epsTensile*FastMath::pow4(Wi/(Hdeti*WnPerh));
      const auto fj = epsTensile*FastMath::pow4(Wj/(Hdetj*WnPerh));
      const auto Ri = fi*tensileStressCorrection(sigmai);
      const auto Rj = fj*tensileStressCorrection(sigmaj);
      sigmai += Ri;
      sigmaj += Rj;

      // Acceleration.
      CHECK(rhoi > 0.0);
      CHECK(rhoj > 0.0);
      const auto sigmarhoi = safeOmegai*sigmai/(rhoi*rhoi);
      const auto sigmarhoj = safeOmegaj*sigmaj/(rhoj*rhoj);
      const auto deltaDvDt = sigmarhoi*gradWi + sigmarhoj*gradWj - Qacci - Qaccj;
      if (freeParticle) {
        DvDti += mj*deltaDvDt;
        DvDtj -= mi*deltaDvDt;
      }
      if (compatibleEnergy) {
        pairAccelerations[2*kk]   =  mj*deltaDvDt;
        pairAccelerations[2*kk+1] = -mi*deltaDvDt;
      }

      // Pair-wise portion of grad velocity.
      const auto deltaDvDxi = fDij * vij.dyad(gradWGi);
      const auto deltaDvDxj = fDij * vij.dyad(gradWGj);

      // Specific thermal energy evolution.
      DepsDti -= mj*(sigmarhoi.doubledot(deltaDvDxi.Symmetric()) - workQi);
      DepsDtj -= mi*(sigmarhoj.doubledot(deltaDvDxj.Symmetric()) - workQj);

      // Velocity gradient.
      DvDxi -= mj*deltaDvDxi;
      DvDxj -= mi*deltaDvDxj;
      if (sameMatij) {
        localDvDxi -= mj*deltaDvDxi;
        localDvDxj -= mi*deltaDvDxj;
      }

      // Estimate of delta v (for XSPH).
      if (XSPH and sameMatij) {
        const auto wXSPHij = 0.5*(mi/rhoi*Wi + mj/rhoj*Wj);
        XSPHWeightSumi += wXSPHij;
        XSPHWeightSumj += wXSPHij;
        XSPHDeltaVi -= wXSPHij*vij;
        XSPHDeltaVj += wXSPHij*vij;
      }

      // Linear gradient correction term.
      Mi -= mj*xij.dyad(gradWGi);
      Mj -= mi*xij.dyad(gradWGj);
      if (sameMatij) {
        localMi -= mj*xij.dyad(gradWGi);
        localMj -= mi*xij.dyad(gradWGj);
      }

    } // loop over pairs

    // Reduce the thread values to the master.
    threadReduceFieldLists<Dimension>(threadStack);

  }   // OpenMP parallel region

  // Finish up the derivatives for each point.
  auto offset = 2*npairs;
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto& nodeList = mass[nodeListi]->nodeList();
    const auto  hmin = nodeList.hmin();
    const auto  hmax = nodeList.hmax();
    const auto  hminratio = nodeList.hminratio();
    const auto  nPerh = nodeList.nodesPerSmoothingScale();

    // Check if we can identify a reference density.
    auto rho0 = 0.0;
    try {
      rho0 = dynamic_cast<const SolidEquationOfState<Dimension>&>(dynamic_cast<const FluidNodeList<Dimension>&>(nodeList).equationOfState()).referenceDensity();
      // cerr << "Setting reference density to " << rho0 << endl;
    } catch(...) {
      // cerr << "BLAGO!" << endl;
    }

    const auto ni = nodeList.numInternalNodes();
#pragma omp parallel for
    for (auto i = 0u; i < ni; ++i) {

      // Get the state for node i.
      const auto& posi = position(nodeListi, i);
      const auto  mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto  rhoi = massDensity(nodeListi, i);
      const auto  Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto& Si = S(nodeListi, i);
      const auto  mui = mu(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      const auto  riInv = safeInv(posi.x());
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& rhoSumi = rhoSum(nodeListi, i);
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
      auto& effViscousPressurei = effViscousPressure(nodeListi, i);
      auto& rhoSumCorrectioni = rhoSumCorrection(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment(nodeListi, i);
      auto& DSDti = DSDt(nodeListi, i);

      // Add the self-contribution to density sum.
      const auto W0 = W(Vector::zero, Vector::zero, Hdeti);
      rhoSumi += mi*W0*Hdeti;

      // Add the self-contribution to density sum correction.
      const auto WQ0 = uniqueWQ ? WQ(Vector::zero, Vector::zero, Hdeti) : W0;
      rhoSumCorrectioni += mi*WQ0*Hdeti/rhoi ;

      // Correct the effective viscous pressure.
      effViscousPressurei /= rhoSumCorrectioni;

      // Finish the acceleration -- self hoop strain.
      const auto deltaDvDti = 3.0*Si(0,0)/rhoi*riInv;
      DvDti[0] += deltaDvDti;
      if (compatibleEnergy) pairAccelerations[offset + i] = deltaDvDti;

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
      XSPHWeightSumi += Hdeti*mi/rhoi*W0;
      CHECK2(XSPHWeightSumi != 0.0, i << " " << XSPHWeightSumi);
      XSPHDeltaVi /= XSPHWeightSumi;
      const auto vri = vi.x();
      DrhoDti = -rhoi*(DvDxi[0] + 2.0*vri*riInv);

      // Finish the specific thermal energy evolution.
      DepsDti += (Si.xx() - 2.0*Pi)/rhoi*2.0*vri*riInv;

      // If needed finish the total energy derivative.
      if (this->mEvolveTotalEnergy) DepsDti = mi*(vi.dot(DvDti) + DepsDti);

      // Complete the moments of the node distribution for use in the ideal H calculation.
      weightedNeighborSumi = Dimension::rootnu(max(0.0, weightedNeighborSumi/(Hdeti*Hdeti)));
      massSecondMomenti /= Hdeti*Hdeti;

      // Determine the position evolution, based on whether we're doing XSPH or not.
      if (XSPH) {
        DxDti = vi + XSPHDeltaVi;
      } else {
        DxDti = vi;
      }

      // The H tensor evolution.
      DHDti = smoothingScaleMethod.smoothingScaleDerivative(Hi,
                                                            posi,
                                                            DvDxi,
                                                            hmin,
                                                            hmax,
                                                            hminratio,
                                                            nPerh);
      Hideali = smoothingScaleMethod.newSmoothingScale(Hi,
                                                       posi,
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

      // Optionally use damage to ramp down stress on damaged material.
      const auto Di = (damageRelieveRubble ? 
                       max(0.0, min(1.0, damage(nodeListi, i).Trace() - 1.0)) :
                       0.0);
      // Hideali = (1.0 - Di)*Hideali + Di*Hfield0(nodeListi, i);
      // DHDti = (1.0 - Di)*DHDti + Di*(Hfield0(nodeListi, i) - Hi)*0.25/dt;

      // We also adjust the density evolution in the presence of damage.
      if (rho0 > 0.0) DrhoDti = (1.0 - Di)*DrhoDti - 0.25/dt*Di*(rhoi - rho0);

      // Determine the deviatoric stress evolution.
      const auto deformation = localDvDxi.Symmetric();
      const auto deformationTT = vi.y()*riInv;
      const auto spin = localDvDxi.SkewSymmetric();
      const auto deviatoricDeformation = deformation - ((deformation.Trace() + deformationTT)/3.0)*SymTensor::one;
      const auto spinCorrection = (spin*Si + Si*spin).Symmetric();
      DSDti = spinCorrection + (2.0*mui)*deviatoricDeformation;

      // In the presence of damage, add a term to reduce the stress on this point.
      DSDti = (1.0 - Di)*DSDti - Di*Si*0.25/dt;
    }
    offset += ni;
  }
}

//------------------------------------------------------------------------------
// Apply the ghost boundary conditions for hydro state fields.
//------------------------------------------------------------------------------
void
SolidSPHHydroBaseR::
applyGhostBoundaries(State<Dim<1>>& state,
                     StateDerivatives<Dim<1>>& derivs) {

  // Convert the mass to mass/length before BCs are applied.
  auto       mass = state.fields(HydroFieldNames::mass, 0.0);
  const auto pos = state.fields(HydroFieldNames::position, Vector::zero);
  const auto numNodeLists = mass.numFields();
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto n = mass[nodeListi]->numElements();
    for (auto i = 0u; i < n; ++i) {
      const auto ri = pos(nodeListi, i).x();
      CHECK(ri > 0.0);
      mass(nodeListi, i) /= ri*ri;
    }
  }

  // Apply ordinary SPH BCs.
  SolidSPHHydroBase<Dim<1>>::applyGhostBoundaries(state, derivs);
  for (auto boundItr = this->boundaryBegin();
       boundItr < this->boundaryEnd();
       ++boundItr) (*boundItr)->finalizeGhostBoundary();

  // Scale back to mass.
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto n = mass[nodeListi]->numElements();
    for (auto i = 0u; i < n; ++i) {
      const auto ri = pos(nodeListi, i).x();
      CHECK(ri > 0.0);
      mass(nodeListi, i) *= ri*ri;
    }
  }
}

//------------------------------------------------------------------------------
// Enforce the boundary conditions for hydro state fields.
//------------------------------------------------------------------------------
void
SolidSPHHydroBaseR::
enforceBoundaries(State<Dim<1>>& state,
                  StateDerivatives<Dim<1>>& derivs) {

  // Convert the mass to mass/length before BCs are applied.
  auto       mass = state.fields(HydroFieldNames::mass, 0.0);
  const auto pos = state.fields(HydroFieldNames::position, Vector::zero);
  const auto numNodeLists = mass.numFields();
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto n = mass[nodeListi]->numElements();
    for (auto i = 0u; i < n; ++i) {
      const auto ri = pos(nodeListi, i).x();
      CHECK(ri > 0.0);
      mass(nodeListi, i) /= ri*ri;
    }
  }

  // Apply ordinary SPH BCs.
  SolidSPHHydroBase<Dim<1>>::enforceBoundaries(state, derivs);

  // Scale back to mass.
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto n = mass[nodeListi]->numElements();
    for (auto i = 0u; i < n; ++i) {
      const auto ri = pos(nodeListi, i).x();
      CHECK(ri > 0.0);
      mass(nodeListi, i) *= ri*ri;
    }
  }
}

//------------------------------------------------------------------------------
// Access teh spherical kernels.
//------------------------------------------------------------------------------
const SphericalTableKernel&
SolidSPHHydroBaseR::
sphericalKernel() const {
  CHECK(mKernelPtr != nullptr);
  return *mKernelPtr;
}

const SphericalTableKernel&
SolidSPHHydroBaseR::
sphericalPiKernel() const {
  CHECK(mPiKernelPtr != nullptr);
  return *mPiKernelPtr;
}

const SphericalTableKernel&
SolidSPHHydroBaseR::
sphericalGradKernel() const {
  CHECK(mGKernelPtr != nullptr);
  return *mGKernelPtr;
}

}
