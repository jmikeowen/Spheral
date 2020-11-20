//---------------------------------Spheral++----------------------------------//
// SolidSPHHydroBase -- SolidSPHHydro modified for large density discontinuities
//
// Created by JMO, Fri Jul 30 11:07:33 PDT 2010
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "Utilities/DamagedNodeCouplingWithFrags.hh"
#include "SPH/SPHHydroBase.hh"
#include "SPH/SolidSPHHydroBase.hh"
#include "NodeList/SmoothingScaleBase.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Strength/SolidFieldNames.hh"
#include "NodeList/SolidNodeList.hh"
#include "Strength/DeviatoricStressPolicy.hh"
#include "Strength/BulkModulusPolicy.hh"
#include "Strength/PlasticStrainPolicy.hh"
#include "Strength/ShearModulusPolicy.hh"
#include "Strength/YieldStrengthPolicy.hh"
#include "Strength/StrengthSoundSpeedPolicy.hh"
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

#include "FSISolidSPHHydroBase.hh"

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


inline
Dim<1>::SymTensor
tensileStressCorrection(const Dim<1>::SymTensor& sigma) {
  if (sigma.xx() > 0.0) {
    return -sigma;
  } else {
    return Dim<1>::SymTensor::zero;
  }
}

inline
Dim<2>::SymTensor
tensileStressCorrection(const Dim<2>::SymTensor& sigma) {
  const EigenStruct<2> eigen = sigma.eigenVectors();
  const double lambdax = eigen.eigenValues.x();
  const double lambday = eigen.eigenValues.y();
  Dim<2>::SymTensor result((lambdax > 0.0 ? -lambdax : 0.0), 0.0,
                           0.0,                              (lambday > 0.0 ? -lambday : 0.0));
  result.rotationalTransform(eigen.eigenVectors);
  return result;
}

inline
Dim<3>::SymTensor
tensileStressCorrection(const Dim<3>::SymTensor& sigma) {
  const EigenStruct<3> eigen = sigma.eigenVectors();
  const double lambdax = eigen.eigenValues.x();
  const double lambday = eigen.eigenValues.y();
  const double lambdaz = eigen.eigenValues.z();
  Dim<3>::SymTensor result((lambdax > 0.0 ? -lambdax : 0.0), 0.0,                              0.0,
                           0.0,                              (lambday > 0.0 ? -lambday : 0.0), 0.0,
                           0.0,                              0.0,                              (lambdaz > 0.0 ? -lambdaz : 0.0));
  result.rotationalTransform(eigen.eigenVectors);
  return result;
}


//------------------------------------------------------------------------------
// Construct with the given artificial viscosity and kernels.
//------------------------------------------------------------------------------
template<typename Dimension>
FSISolidSPHHydroBase<Dimension>::
FSISolidSPHHydroBase(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
                  DataBase<Dimension>& dataBase,
                  ArtificialViscosity<Dimension>& Q,
                  const TableKernel<Dimension>& W,
                  const TableKernel<Dimension>& WPi,
                  const TableKernel<Dimension>& WGrad,
                  const double generalizedExponent,
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
  SolidSPHHydroBase<Dimension>(smoothingScaleMethod,
                               dataBase,
                               Q,
                               W,
                               WPi,
                               WGrad,
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
                               xmax),
  mAlpha(generalizedExponent){

  }

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
FSISolidSPHHydroBase<Dimension>::
~FSISolidSPHHydroBase() {
}


//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
FSISolidSPHHydroBase<Dimension>::
evaluateDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar dt,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivatives) const {


  // Get the ArtificialViscosity.
  auto& Q = this->artificialViscosity();

  // The kernels and such.
  const auto& W = this->kernel();
  const auto& WQ = this->PiKernel();
  const auto& WG = this->GradKernel();
  const auto& smoothingScaleMethod = this->smoothingScaleMethod();

  // A few useful constants we'll use in the following loop.
  const auto tiny = 1.0e-30;
  const auto W0 = W(0.0, 1.0);
  const auto WQ0 = WQ(0.0, 1.0);
  const auto epsTensile = this->epsilonTensile();
  const auto compatibleEnergy = this->compatibleEnergyEvolution();
  const auto XSPH = this->XSPH();
  const auto damageRelieveRubble = this->damageRelieveRubble();
  const auto strengthInDamage = this->strengthInDamage();
  const auto negativePressureInDamage = this->negativePressureInDamage();

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
  const auto damage = state.fields(SolidFieldNames::effectiveTensorDamage, SymTensor::zero);
  const auto gradDamage = state.fields(SolidFieldNames::damageGradient, Vector::zero);
  const auto fragIDs = state.fields(SolidFieldNames::fragmentIDs, int(1));
  const auto pTypes = state.fields(SolidFieldNames::particleTypes, int(0));
  const auto K = state.fields(SolidFieldNames::bulkModulus, 0.0);

  typedef typename Dimension::Scalar Scalar; //10/14/2020
  FieldList<Dimension, Scalar> wsum(FieldStorageType::CopyFields);  //10/14/2020
    for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    wsum.appendNewField("sumUnity", massDensity[nodeListi]->nodeList(), 0.0);
    }
    
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
  CHECK(gradDamage.size() == numNodeLists);
  CHECK(fragIDs.size() == numNodeLists);
  CHECK(pTypes.size() == numNodeLists);
  CHECK(K.size() == numNodeLists);

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

  // Size up the pair-wise accelerations before we start.
  if (compatibleEnergy) pairAccelerations.resize(npairs);

  // The scale for the tensile correction.
  const auto& nodeList = mass[0]->nodeList();
  const auto  nPerh = nodeList.nodesPerSmoothingScale();
  const auto  WnPerh = W(1.0/nPerh, 1.0);

  // Build the functor we use to compute the effective coupling between nodes.
  DamagedNodeCouplingWithFrags<Dimension> coupling(damage, gradDamage, H, fragIDs);


  // Walk all the interacting pairs.
#pragma omp parallel
  {
    // Thread private  scratch variables.
    int i, j, nodeListi, nodeListj;
    Scalar Wi, gWi, WQi, gWQi, Wj, gWj, WQj, gWQj;
    //Tensor QPiij, QPiji;
    //SymTensor sigmai, sigmaj;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;
    //auto wsum_thread = wsum.threadCopy(threadStack);
    //auto rhoSum_thread = rhoSum.threadCopy(threadStack);
    //auto DvDt_thread = DvDt.threadCopy(threadStack);
    //auto DepsDt_thread = DepsDt.threadCopy(threadStack);
    auto DvDx_thread = DvDx.threadCopy(threadStack);
    auto localDvDx_thread = localDvDx.threadCopy(threadStack);
    auto M_thread = M.threadCopy(threadStack);
    auto localM_thread = localM.threadCopy(threadStack);
    //auto maxViscousPressure_thread = maxViscousPressure.threadCopy(threadStack, ThreadReduction::MAX);
    //auto effViscousPressure_thread = effViscousPressure.threadCopy(threadStack);
    //auto rhoSumCorrection_thread = rhoSumCorrection.threadCopy(threadStack);
    //auto viscousWork_thread = viscousWork.threadCopy(threadStack);
    auto XSPHWeightSum_thread = XSPHWeightSum.threadCopy(threadStack);
    auto XSPHDeltaV_thread = XSPHDeltaV.threadCopy(threadStack);
    //auto weightedNeighborSum_thread = weightedNeighborSum.threadCopy(threadStack);
    //auto massSecondMoment_thread = massSecondMoment.threadCopy(threadStack);
    //auto DSDt_thread = DSDt.threadCopy(threadStack);
    //auto DrhoDt_thread = DrhoDt.threadCopy(threadStack);

#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      const auto start = Timing::currentTime();
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto  mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto  rhoi = massDensity(nodeListi, i);
      //const auto  epsi = specificThermalEnergy(nodeListi, i);
      //const auto  Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      //const auto  ci = soundSpeed(nodeListi, i);
      //const auto  omegai = omega(nodeListi, i);
      //const auto& Si = S(nodeListi, i);
      //const auto  mui = mu(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      //const auto  safeOmegai = safeInv(omegai, tiny);
      //const auto  fragIDi = fragIDs(nodeListi, i);
      //const auto  pTypei = pTypes(nodeListi, i);
      const auto Ki = K(nodeListi,i);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      //auto& wsumi = wsum_thread(nodeListi, i);
      //auto& DrhoDti = DrhoDt_thread(nodeListi, i);
      //auto& rhoSumi = rhoSum_thread(nodeListi, i);
      //auto& DvDti = DvDt_thread(nodeListi, i);
      //auto& DepsDti = DepsDt_thread(nodeListi, i);
      auto& DvDxi = DvDx_thread(nodeListi, i);
      auto& localDvDxi = localDvDx_thread(nodeListi, i);
      auto& Mi = M_thread(nodeListi, i);
      auto& localMi = localM_thread(nodeListi, i);
      //auto& maxViscousPressurei = maxViscousPressure_thread(nodeListi, i);
      //auto& effViscousPressurei = effViscousPressure_thread(nodeListi, i);
      //auto& rhoSumCorrectioni = rhoSumCorrection_thread(nodeListi, i);
      //auto& viscousWorki = viscousWork_thread(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum_thread(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV_thread(nodeListi, i);
      //auto& weightedNeighborSumi = weightedNeighborSum_thread(nodeListi, i);
      //auto& massSecondMomenti = massSecondMoment_thread(nodeListi, i);

      // Get the state for node j
      const auto rj = position(nodeListj, j);
      const auto mj = mass(nodeListj, j);
      const auto vj = velocity(nodeListj, j);
      const auto rhoj = massDensity(nodeListj, j);
      //const auto epsj = specificThermalEnergy(nodeListj, j);
      //const auto Pj = pressure(nodeListj, j);
      const auto Hj = H(nodeListj, j);
      //const auto cj = soundSpeed(nodeListj, j);
      //const auto omegaj = omega(nodeListj, j);
      //const auto Sj = S(nodeListj, j);
      const auto Hdetj = Hj.Determinant();
      //const auto safeOmegaj = safeInv(omegaj, tiny);
      //const auto fragIDj = fragIDs(nodeListj, j);
      //const auto pTypej = pTypes(nodeListj, j);
      const auto Kj = K(nodeListj,j);
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);


      auto& DvDxj = DvDx_thread(nodeListj, j);
      auto& localDvDxj = localDvDx_thread(nodeListj, j);
      auto& Mj = M_thread(nodeListj, j);
      auto& localMj = localM_thread(nodeListj, j);
      auto& XSPHWeightSumj = XSPHWeightSum_thread(nodeListj, j);
      auto& XSPHDeltaVj = XSPHDeltaV_thread(nodeListj, j);

      // Flag if this is a contiguous material pair or not.
      const auto sameMatij =  (nodeListi == nodeListj);

      // Node displacement.
      const auto rij = ri - rj;
      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);

      // Symmetrized kernel weight and gradient.
      //std::tie(Wi, gWi) = W.kernelAndGradValue(etaMagi, Hdeti);
      //std::tie(WQi, gWQi) = WQ.kernelAndGradValue(etaMagi, Hdeti);
      const auto Hetai = Hi*etai.unitVector();
      //const auto gradWi = gWi*Hetai;
      //const auto gradWQi = gWQi*Hetai;
      const auto gradWGi = WG.gradValue(etaMagi, Hdeti) * Hetai;

      //std::tie(Wj, gWj) = W.kernelAndGradValue(etaMagj, Hdetj);
      //std::tie(WQj, gWQj) = WQ.kernelAndGradValue(etaMagj, Hdetj);
      const auto Hetaj = Hj*etaj.unitVector();
      //const auto gradWj = gWj*Hetaj;
      //const auto gradWQj = gWQj*Hetaj;
      const auto gradWGj = WG.gradValue(etaMagj, Hdetj) * Hetaj;

      // Compute the pair-wise artificial viscosity.
      const auto vij = vi - vj;
      //const auto rhoij = 0.5*(rhoi+rhoj); //pow(rhoi*rhoj,0.5);//11/3/2020
      //const auto cij = 0.5*(ci+cj);      //11/3/2020

      auto kappai = 1.0;
      auto kappaj = 1.0;
      auto kappaMax = 1.0;

      if(!sameMatij){
        kappai = 1.0*(Kj)/(Ki+Kj);
        kappaj = max(0.0,min(1.0,(1.0-kappai)));
      }
      
      const auto voli = mi/rhoi;
      const auto volj = mj/rhoj;
      //const auto deltaDvDxi = vij.dyad(gradWGi);
      //const auto deltaDvDxj = vij.dyad(gradWGj);
  

      //DvDxi -= volj*deltaDvDxi;
      //DvDxj -= voli*deltaDvDxj;
      //if (sameMatij) {
      //  localDvDxi -= volj*deltaDvDxi;
      //  localDvDxj -= voli*deltaDvDxj;
      //
      
      // Linear gradient correction term.
      Mi -= volj*rij.dyad(gradWGi);
      Mj -= voli*rij.dyad(gradWGj);
      if (sameMatij) {
        localMi -= volj*rij.dyad(gradWGi);
        localMj -= voli*rij.dyad(gradWGj);
      }
      
      if (XSPH) {
          //const auto wXSPHij = 0.5*(mi/rhoi*Wi + mj/rhoj*Wj);
          XSPHWeightSumi += kappai*volj*Wi;
          XSPHWeightSumj += kappaj*voli*Wj;
          XSPHDeltaVi -= kappai*vij*volj*Wi;//wXSPHij*vij;
          XSPHDeltaVj += kappaj*vij*voli*Wj;//wXSPHij*vij;
      }

      // Add timing info for work
      const auto deltaTimePair = 0.5*Timing::difference(start, Timing::currentTime());
#pragma omp atomic
      nodeLists[nodeListi]->work()(i) += deltaTimePair;
#pragma omp atomic
      nodeLists[nodeListj]->work()(j) += deltaTimePair;

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
      //const auto& ri = position(nodeListi, i);
      //const auto& mi = mass(nodeListi, i);
      //const auto& vi = velocity(nodeListi, i);
      //const auto& rhoi = massDensity(nodeListi, i);
      //const auto& Hi = H(nodeListi, i);
      //const auto& Si = S(nodeListi, i);
      //const auto& mui = mu(nodeListi, i);
      //const auto  Hdeti = Hi.Determinant();
      const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      //const auto omegai = omega(nodeListi, i);
      //const auto safeOmegai = safeInv(omegai, tiny);
      //CHECK(mi > 0.0);
      //CHECK(rhoi > 0.0);
      // CHECK(Hdeti > 0.0);

      //auto& rhoSumi = rhoSum(nodeListi, i);
      //auto& DxDti = DxDt(nodeListi, i);
      //auto& DrhoDti = DrhoDt(nodeListi, i);
      //auto& DvDti = DvDt(nodeListi, i);
      //auto& DepsDti = DepsDt(nodeListi, i);
      auto& DvDxi = DvDx(nodeListi, i);
      auto& localDvDxi = localDvDx(nodeListi, i);
      auto& Mi = M(nodeListi, i);
      auto& localMi = localM(nodeListi, i);
      //auto& DHDti = DHDt(nodeListi, i);
      //auto& Hideali = Hideal(nodeListi, i);
      //auto& effViscousPressurei = effViscousPressure(nodeListi, i);
      //auto& rhoSumCorrectioni = rhoSumCorrection(nodeListi, i);
      //auto& XSPHWeightSumi = XSPHWeightSum(nodeListi, i);
      //auto& XSPHDeltaVi = XSPHDeltaV(nodeListi, i);
      //auto& weightedNeighborSumi = weightedNeighborSum(nodeListi, i);
      //auto& massSecondMomenti = massSecondMoment(nodeListi, i);
      //auto& DSDti = DSDt(nodeListi, i);

      // Finish the gradient of the velocity.
      if (this->mCorrectVelocityGradient and
          std::abs(Mi.Determinant()) > 1.0e-10 and
          numNeighborsi > Dimension::pownu(2)) {
        Mi = Mi.Inverse();
      } else {  //11/05/2020
        Mi = Tensor::one;  //11/05/2020
      }  //11/05/2020
      if (this->mCorrectVelocityGradient and
          std::abs(localMi.Determinant()) > 1.0e-10 and
          numNeighborsi > Dimension::pownu(2)) {
          localMi = localMi.Inverse();
      } else {  //11/05/2020
        localMi = Tensor::one;  //11/05/2020
      }

    }
  }


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// original!!!!!!!!!!!!!!!!!!!!!!!
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // Walk all the interacting pairs.
#pragma omp parallel
  {
    // Thread private  scratch variables.
    int i, j, nodeListi, nodeListj;
    Scalar Wi, gWi, WQi, gWQi, Wj, gWj, WQj, gWQj;
    Tensor QPiij, QPiji;
    SymTensor sigmai, sigmaj;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;
    auto wsum_thread = wsum.threadCopy(threadStack);
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
    auto DrhoDt_thread = DrhoDt.threadCopy(threadStack);

#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      const auto start = Timing::currentTime();
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto  mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto  rhoi = massDensity(nodeListi, i);
      //const auto  epsi = specificThermalEnergy(nodeListi, i);
      const auto  Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  ci = soundSpeed(nodeListi, i);
      const auto  omegai = omega(nodeListi, i);
      const auto& Si = S(nodeListi, i);
      //const auto  mui = mu(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      const auto  safeOmegai = safeInv(omegai, tiny);
      //const auto  fragIDi = fragIDs(nodeListi, i);
      const auto  pTypei = pTypes(nodeListi, i);
      const auto Ki = K(nodeListi,i);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& wsumi = wsum_thread(nodeListi, i);
      auto& DrhoDti = DrhoDt_thread(nodeListi, i);
      auto& rhoSumi = rhoSum_thread(nodeListi, i);
      auto& DvDti = DvDt_thread(nodeListi, i);
      auto& DepsDti = DepsDt_thread(nodeListi, i);
      auto& DvDxi = DvDx_thread(nodeListi, i);
      auto& localDvDxi = localDvDx_thread(nodeListi, i);
      const auto Mi = M_thread(nodeListi, i);
      const auto localMi = localM_thread(nodeListi, i);
      auto& maxViscousPressurei = maxViscousPressure_thread(nodeListi, i);
      auto& effViscousPressurei = effViscousPressure_thread(nodeListi, i);
      auto& rhoSumCorrectioni = rhoSumCorrection_thread(nodeListi, i);
      auto& viscousWorki = viscousWork_thread(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum_thread(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV_thread(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum_thread(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment_thread(nodeListi, i);

      // Get the state for node j
      const auto rj = position(nodeListj, j);
      const auto mj = mass(nodeListj, j);
      const auto vj = velocity(nodeListj, j);
      const auto rhoj = massDensity(nodeListj, j);
      //const auto epsj = specificThermalEnergy(nodeListj, j);
      const auto Pj = pressure(nodeListj, j);
      const auto Hj = H(nodeListj, j);
      const auto cj = soundSpeed(nodeListj, j);
      const auto omegaj = omega(nodeListj, j);
      const auto Sj = S(nodeListj, j);
      const auto Hdetj = Hj.Determinant();
      const auto safeOmegaj = safeInv(omegaj, tiny);
      //const auto fragIDj = fragIDs(nodeListj, j);
      const auto pTypej = pTypes(nodeListj, j);
      const auto Kj = K(nodeListj,j);
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& wsumj = wsum_thread(nodeListj, j);
      auto& DrhoDtj = DrhoDt_thread(nodeListj, j);
      auto& rhoSumj = rhoSum_thread(nodeListj, j);
      auto& DvDtj = DvDt_thread(nodeListj, j);
      auto& DepsDtj = DepsDt_thread(nodeListj, j);
      auto& DvDxj = DvDx_thread(nodeListj, j);
      auto& localDvDxj = localDvDx_thread(nodeListj, j);
      const auto Mj = M_thread(nodeListj, j);
      const auto localMj = localM_thread(nodeListj, j);
      auto& maxViscousPressurej = maxViscousPressure_thread(nodeListj, j);
      auto& effViscousPressurej = effViscousPressure_thread(nodeListj, j);
      auto& rhoSumCorrectionj = rhoSumCorrection_thread(nodeListj, j);
      auto& viscousWorkj = viscousWork_thread(nodeListj, j);
      auto& XSPHWeightSumj = XSPHWeightSum_thread(nodeListj, j);
      auto& XSPHDeltaVj = XSPHDeltaV_thread(nodeListj, j);
      auto& weightedNeighborSumj = weightedNeighborSum_thread(nodeListj, j);
      auto& massSecondMomentj = massSecondMoment_thread(nodeListj, j);

      // Flag if this is a contiguous material pair or not.
      const auto sameMatij =  (nodeListi == nodeListj);// and fragIDi == fragIDj);  //11/05/2020

      // Flag if at least one particle is free (0).
      const auto freeParticle = (pTypei == 0 or pTypej == 0);

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
      const auto gradWGi = WG.gradValue(etaMagi, Hdeti) * Hetai;

      std::tie(Wj, gWj) = W.kernelAndGradValue(etaMagj, Hdetj);
      std::tie(WQj, gWQj) = WQ.kernelAndGradValue(etaMagj, Hdetj);
      const auto Hetaj = Hj*etaj.unitVector();
      const auto gradWj = gWj*Hetaj;
      const auto gradWQj = gWQj*Hetaj;
      const auto gradWGj = WG.gradValue(etaMagj, Hdetj) * Hetaj;

      // Determine how we're applying damage.
      const auto fDeffij = coupling(nodeListi, i, nodeListj, j);

      // Zero'th and second moment of the node distribution -- used for the
      // ideal H calculation.
      const auto fweightij = 1.0;//sameMatij ? 1.0 : mj*rhoi/(mi*rhoj); //11/05/2020
      const auto rij2 = rij.magnitude2();
      const auto thpt = rij.selfdyad()*safeInvVar(rij2*rij2*rij2);
      weightedNeighborSumi +=     fweightij*abs(gWi);
      weightedNeighborSumj += 1.0/fweightij*abs(gWj);
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

      // bulk modulus is used to treat multiple materials
      

      // Compute the pair-wise artificial viscosity.
      const auto vij = vi - vj;
      const auto rhoij = 0.5*(rhoi+rhoj); //pow(rhoi*rhoj,0.5);//11/3/2020
      const auto cij = 0.5*(ci+cj);      //11/3/2020

      auto kappai = 1.0;
      auto kappaj = 1.0;
      auto kappaMax = 1.0;

      if(!sameMatij){
        kappai = 1.0*(Kj)/(Ki+Kj);
        kappaj = max(0.0,min(1.0,(1.0-kappai)));
        kappaMax = 1.0*max(kappai,kappaj);
      }
      
      std::tie(QPiij, QPiji) = Q.Piij(nodeListi, i, nodeListj, j,
                                      ri, etai, vi, rhoij, cij, Hi,  
                                      rj, etaj, vj, rhoij, cij, Hj); //11/3/2020

      //auto Qacci = 0.5*(QPiij*gradWGi);
      //auto Qaccj = 0.5*(QPiji*gradWGj);
      //auto workQi = kappai*0.5*(QPiij.doubledot(vij.dyad(gradWGi)));
      //auto workQj = kappaj*0.5*(QPiji.doubledot(vij.dyad(gradWGj)));
      //const auto Qi = rhoi*rhoi*(QPiij.diagonalElements().maxAbsElement());
      //const auto Qj = rhoj*rhoj*(QPiji.diagonalElements().maxAbsElement());
      //maxViscousPressurei = max(maxViscousPressurei, Qi);
      //maxViscousPressurej = max(maxViscousPressurej, Qj);
      //effViscousPressurei += mj*Qi*WQi/rhoj;
      //effViscousPressurej += mi*Qj*WQj/rhoi;
      //viscousWorki += mj*workQi;
      //viscousWorkj += mi*workQj;

      // Damage scaling of negative pressures.
  
      auto Peffi = (negativePressureInDamage or Pi > 0.0 ? Pi : fDeffij*Pi);
      auto Peffj = (negativePressureInDamage or Pj > 0.0 ? Pj : fDeffij*Pj);
      if (!sameMatij){
        Peffi = ( Pi > 0.0 ? Pi : 0.0*Pi);
        Peffj = ( Pj > 0.0 ? Pj : 0.0*Pj);
      }
      //if (sameMatij)
      // Compute the stress tensors.
      sigmai = -Peffi*SymTensor::one;
      sigmaj = -Peffj*SymTensor::one;
      if (sameMatij) {
        if (strengthInDamage) {
          sigmai += Si;
          sigmaj += Sj;
        } else {
          sigmai += fDeffij*Si;
          sigmaj += fDeffij*Sj;
        }
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
      const auto alpha = 1.1;
      const auto voli = mi/rhoi;
      const auto volj = mj/rhoj;
      const auto oneOverRhoiRhoj = 1.0/(rhoi*rhoj);
      const auto sigmarhoi = sigmai/(pow(rhoi,alpha)*pow(rhoj,2.0-alpha))-0.5*(QPiij);//-0.5*QPiij;
      const auto sigmarhoj = sigmaj/(pow(rhoj,alpha)*pow(rhoi,2.0-alpha))-0.5*(QPiji);//-0.5*QPiji;
      const auto deltaDvDt = safeOmegai*sigmarhoi*gradWi + safeOmegaj*sigmarhoj*gradWj;//-Qacci-Qaccj;
      //const auto deltaDvDti = ((sigmarhoj-sigmarhoi)*gradWi);//-Qacci-Qaccj;
      //const auto deltaDvDtj = ((sigmarhoi-sigmarhoj)*gradWj);//-Qacci-Qaccj;
      //const auto deltaDvDt = oneOverRhoiRhoj*(sigmaj - sigmai)*(gradWj+gradWi)/2.0 - Qacci - Qaccj;
      if (freeParticle) {
        DvDti += mj*deltaDvDt;
        DvDtj -= mi*deltaDvDt;
      }
      if (compatibleEnergy) pairAccelerations[kk] = mj*deltaDvDt;  // Acceleration for i (j anti-symmetric)

      const auto deltaDvDxi = vij.dyad(gradWGi);
      const auto deltaDvDxj = vij.dyad(gradWGj);
      // Velocity gradient.

      DvDxi -= kappai*mj/rhoj*(deltaDvDxi)*Mi;
      DvDxj -= kappaj*mi/rhoi*(deltaDvDxj)*Mj;
      if (sameMatij) {
        localDvDxi -= kappai*mj/rhoj*(deltaDvDxi)*localMi;
        localDvDxj -= kappaj*mi/rhoi*(deltaDvDxj)*localMj;
      }

      //bulk modulus correction only applies to the diagonal elements
      //omega correction invalid of we are doing the stitching this way
      DepsDti -= mj*(sigmarhoi.doubledot(DvDxi));//fVoli*deltaDepsDt/mi;//*sigmarhoi.doubledot(deltaDvDxi.Symmetric()) - workQi );//mj*deltaDepsDt;
      DepsDtj -= mi*(sigmarhoj.doubledot(DvDxj));//fVolj*deltaDepsDt/mj;//*sigmarhoj.doubledot(deltaDvDxj.Symmetric()) - workQj );//


      // Add timing info for work
      const auto deltaTimePair = 0.5*Timing::difference(start, Timing::currentTime());
#pragma omp atomic
      nodeLists[nodeListi]->work()(i) += deltaTimePair;
#pragma omp atomic
      nodeLists[nodeListj]->work()(j) += deltaTimePair;

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
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto& Si = S(nodeListi, i);
      const auto& mui = mu(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      const auto omegai = omega(nodeListi, i);
      const auto safeOmegai = safeInv(omegai, tiny);
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
      rhoSumi += mi*W0*Hdeti;

      // Add the self-contribution to density sum correction.
      rhoSumCorrectioni += mi*WQ0*Hdeti/rhoi ;

      // Correct the effective viscous pressure.
      effViscousPressurei /= rhoSumCorrectioni ;
      
      // If needed finish the total energy derivative.
      if (this->mEvolveTotalEnergy) DepsDti = mi*(vi.dot(DvDti) + DepsDti);

      
      // Complete the moments of the node distribution for use in the ideal H calculation.
      weightedNeighborSumi = Dimension::rootnu(max(0.0, weightedNeighborSumi/Hdeti));
      massSecondMomenti /= Hdeti*Hdeti;

            // Finish the gradient of the velocity.
      //CHECK(rhoi > 0.0);
      //if (this->mCorrectVelocityGradient and
      //    std::abs(Mi.Determinant()) > 1.0e-10 and
      //    numNeighborsi > Dimension::pownu(2)) {
      //  Mi = Mi.Inverse();
      //  DvDxi = DvDxi*Mi;
      //} //else {
        //DvDxi /= rhoi;
      //}
      //if (this->mCorrectVelocityGradient and
      //    std::abs(localMi.Determinant()) > 1.0e-10 and
      //    numNeighborsi > Dimension::pownu(2)) {
      //  localMi = localMi.Inverse();
      //  localDvDxi = localDvDxi*localMi;
      //} //else {
        //localDvDxi /= rhoi;
      //}

      // Evaluate the continuity equation.
      //DrhoDti = -rhoi*DvDxi.Trace();
      // Evaluate the continuity equation.
      //if (nodeListi==0){ 
      DrhoDti = -rhoi*DvDxi.Trace();
      //}else{  
      //  DrhoDti = -rhoi*localDvDxi.Trace();            
      //  DvDxi=1.0*localDvDxi;
      //} 
      // Determine the position evolution, based on whether we're doing XSPH or not.
      DxDti = vi;
      if (XSPH) {
        CHECK(XSPHWeightSumi >= 0.0);
        XSPHWeightSumi += Hdeti*mi/rhoi*W0 + 1.0e-30;
        DxDti += XSPHDeltaVi/XSPHWeightSumi;
      }

      // The H tensor evolution.
      DHDti = smoothingScaleMethod.smoothingScaleDerivative(Hi,
                                                            ri,
                                                            DvDxi,
                                                            hmin,
                                                            hmax,
                                                            hminratio,
                                                            nPerh);
      Hideali = smoothingScaleMethod.newSmoothingScale(Hi,
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

      // Optionally use damage to ramp down stress on damaged material.
      const auto Di = (damageRelieveRubble ? 
                       max(0.0, min(1.0, damage(nodeListi, i).Trace() - 1.0)) :
                       0.0);
      // Hideali = (1.0 - Di)*Hideali + Di*mHfield0(nodeListi, i);
      // DHDti = (1.0 - Di)*DHDti + Di*(mHfield0(nodeListi, i) - Hi)*0.25/dt;

      // We also adjust the density evolution in the presence of damage.
      if (rho0 > 0.0) DrhoDti = (1.0 - Di)*DrhoDti - 0.25/dt*Di*(rhoi - rho0);

      // Determine the deviatoric stress evolution.
      const auto deformation = localDvDxi.Symmetric();
      const auto spin = localDvDxi.SkewSymmetric();
      const auto deviatoricDeformation = deformation - (deformation.Trace()/3.0)*SymTensor::one;
      const auto spinCorrection = (spin*Si + Si*spin).Symmetric();
      DSDti = spinCorrection + (2.0*mui)*deviatoricDeformation;

      // In the presence of damage, add a term to reduce the stress on this point.
      DSDti = (1.0 - Di)*DSDti - 0.25/dt*Di*Si;
    }
  }
}

}
