//---------------------------------Spheral++----------------------------------//
// SPHHydroBase -- modified SPHHydro for large density discontinuities
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "computeSPHSumMassDensity.hh"
#include "correctSPHSumMassDensity.hh"
#include "computeSumVoronoiCellMassDensity.hh"
//#include "computeSPHOmegaGradhCorrection.hh"
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

#include "SPH/FSISPHHydroBase.hh"

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
FSISPHHydroBase<Dimension>::
FSISPHHydroBase(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
             DataBase<Dimension>& dataBase,
             ArtificialViscosity<Dimension>& Q,
             const TableKernel<Dimension>& W,
             const TableKernel<Dimension>& WPi,
             const double alpha,
             const double diffusionCoefficient,
             const std::vector<int> sumDensityNodeLists,
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
                          gradhCorrection,
                          XSPH,
                          correctVelocityGradient,
                          sumMassDensityOverAllNodeLists,
                          densityUpdate,
                          HUpdate,
                          epsTensile,
                          nTensile,
                          xmin,
                          xmax),
  mAlpha(alpha),
  mDiffusionCoefficient(diffusionCoefficient),
  mSumDensityNodeLists(sumDensityNodeLists){
  }

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
FSISPHHydroBase<Dimension>::
~FSISPHHydroBase() {
}


//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
FSISPHHydroBase<Dimension>::
evaluateDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar /*dt*/,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivatives) const {
  TIME_SPHevalDerivs.start();
  TIME_SPHevalDerivs_initial.start();

  //static double totalLoopTime = 0.0;
  // Get the ArtificialViscosity.
  auto& Q = this->artificialViscosity();

  // The kernels and such.
  const auto& W = this->kernel();
  const auto& WQ = this->PiKernel();

  // A few useful constants we'll use in the following loop.
  const double tiny = 1.0e-30;
  const Scalar W0 = W(0.0, 1.0);
  const auto alpha = this->alpha();
  const auto diffCoeff = this->diffusionCoefficient();
  const auto evolveTotalEnergy = this->evolveTotalEnergy();
  const auto epsTensile = this->epsilonTensile();
  const auto compatibleEnergyEvolution = this->compatibleEnergyEvolution();
  const auto XSPH = this->XSPH();
  const auto& smoothingScaleMethod = this->smoothingScaleMethod();

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
  //const auto omega = state.fields(HydroFieldNames::omegaGradh, 0.0);
  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(velocity.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(specificThermalEnergy.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);
  CHECK(pressure.size() == numNodeLists);
  CHECK(soundSpeed.size() == numNodeLists);
  //CHECK(omega.size() == numNodeLists);

  // Derivative FieldLists.
  auto  rhoSum = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto  normalization = derivatives.fields(HydroFieldNames::normalization, 0.0);
  auto  DxDt = derivatives.fields(IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, Vector::zero);
  auto  DrhoDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto  DvDt = derivatives.fields(HydroFieldNames::hydroAcceleration, Vector::zero);
  auto  DepsDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy, 0.0);
  auto  DvDx = derivatives.fields(HydroFieldNames::velocityGradient, Tensor::zero);
  //auto  localDvDx = derivatives.fields(HydroFieldNames::internalVelocityGradient, Tensor::zero);
  auto  M = derivatives.fields(HydroFieldNames::M_SPHCorrection, Tensor::zero);
  //auto  localM = derivatives.fields("local " + HydroFieldNames::M_SPHCorrection, Tensor::zero);
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
  //CHECK(localDvDx.size() == numNodeLists);
  CHECK(M.size() == numNodeLists);
  //CHECK(localM.size() == numNodeLists);
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
  if (compatibleEnergyEvolution) pairAccelerations.resize(npairs);

  // The scale for the tensile correction.
  const auto& nodeList = mass[0]->nodeList();
  const auto  nPerh = nodeList.nodesPerSmoothingScale();
  const auto  WnPerh = W(1.0/nPerh, 1.0);
  TIME_SPHevalDerivs_initial.stop();

  // Walk all the interacting pairs.
  TIME_SPHevalDerivs_pairs.start();
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
    auto DrhoDt_thread = DrhoDt.threadCopy(threadStack);
    auto DvDx_thread = DvDx.threadCopy(threadStack);
    //auto localDvDx_thread = localDvDx.threadCopy(threadStack);
    auto M_thread = M.threadCopy(threadStack);
    //auto localM_thread = localM.threadCopy(threadStack);
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
      const auto& epsi = specificThermalEnergy(nodeListi,i);
      const auto& Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto& ci = soundSpeed(nodeListi, i);
      //const auto& omegai = omega(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      //const auto  safeOmegai = safeInv(omegai, tiny);
      const auto Ki = max(tiny,rhoi*ci*ci);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& rhoSumi = rhoSum_thread(nodeListi, i);
      auto& normi = normalization_thread(nodeListi, i);
      auto& DvDti = DvDt_thread(nodeListi, i);
      auto& DepsDti = DepsDt_thread(nodeListi, i);
      auto& DrhoDti = DrhoDt_thread(nodeListi, i);
      auto& DvDxi = DvDx_thread(nodeListi, i);
      //auto& localDvDxi = localDvDx_thread(nodeListi, i);
      auto& Mi = M_thread(nodeListi, i);
      //auto& localMi = localM_thread(nodeListi, i);
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
      const auto& epsj = specificThermalEnergy(nodeListj,j);
      const auto& Pj = pressure(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto& cj = soundSpeed(nodeListj, j);
      //const auto& omegaj = omega(nodeListj, j);
      const auto  Hdetj = Hj.Determinant();
      //const auto  safeOmegaj = safeInv(omegaj, tiny); 
      const auto Kj = max(tiny,rhoj*cj*cj);
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& rhoSumj = rhoSum_thread(nodeListj, j);
      auto& normj = normalization_thread(nodeListj, j);
      auto& DvDtj = DvDt_thread(nodeListj, j);
      auto& DepsDtj = DepsDt_thread(nodeListj, j);
      auto& DrhoDtj = DrhoDt_thread(nodeListj, j);
      auto& DvDxj = DvDx_thread(nodeListj, j);
      //auto& localDvDxj = localDvDx_thread(nodeListj, j);
      auto& Mj = M_thread(nodeListj, j);
      //auto& localMj = localM_thread(nodeListj, j);
      auto& maxViscousPressurej = maxViscousPressure_thread(nodeListj, j);
      auto& effViscousPressurej = effViscousPressure_thread(nodeListj, j);
      auto& viscousWorkj = viscousWork_thread(nodeListj, j);
      auto& XSPHWeightSumj = XSPHWeightSum_thread(nodeListj, j);
      auto& XSPHDeltaVj = XSPHDeltaV_thread(nodeListj, j);
      auto& weightedNeighborSumj = weightedNeighborSum_thread(nodeListj, j);
      auto& massSecondMomentj = massSecondMoment_thread(nodeListj, j);

      // Flag if this is a contiguous material pair or not.
      const bool sameMatij =  (nodeListi == nodeListj);

      // Node displacement.
      const auto rij = ri - rj;
      const auto Hij = 0.5*(Hi+Hj);
      const auto etaij = Hij*rij;
      const auto Hdetij = Hij.Determinant();
      const auto etaMagij = etaij.magnitude();
      CHECK(etaMagij >= 0.0);

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
      if (sameMatij) {
        rhoSumi += mj*Wi;
        rhoSumj += mi*Wj;
        normi += mi/rhoi*Wi;
        normj += mj/rhoj*Wj;
      }

      // averaged things.
      auto vij = vi - vj;
      const auto rhoij = 0.5*(rhoi+rhoj); 
      const auto cij = 0.5*(ci+cj);  
      const auto Wij = 0.5*(Wi+Wj); 
      const auto gWij = 0.5*(gWi+gWj);
      const auto gradWij = 0.5*(gradWi+gradWj);
       
      // volumes
      const auto voli = mi/rhoi;
      const auto volj = mj/rhoj;

      std::tie(QPiij, QPiji) = Q.Piij(nodeListi, i, nodeListj, j,
                                      ri, etaij, vi, rhoij, cij, Hij,  
                                      rj, etaij, vj, rhoij, cij, Hij); 

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

      // Determine an effective pressure including a term to fight the tensile instability.
//             const auto fij = epsTensile*pow(Wi/(Hdeti*WnPerh), nTensile);
      const auto fij = epsTensile*FastMath::pow4(Wi/(Hdeti*WnPerh));
      const auto Ri = fij*(Pi < 0.0 ? -Pi : 0.0);
      const auto Rj = fij*(Pj < 0.0 ? -Pj : 0.0);
      const auto Peffi = Pi + Ri;
      const auto Peffj = Pj + Rj;

      // Acceleration.
      CHECK(rhoi > 0.0);
      CHECK(rhoj > 0.0);
      const auto oneOverRhoiRhoj = 1.0/(rhoi*rhoj);
      const auto Pstar = (Peffi*rhoj+Peffj*rhoi)/(rhoi+rhoj);
      const auto sf =  1.0 + alpha*abs((rhoi-rhoj)/(rhoi+rhoj));
      const auto Prhoi = (sameMatij ? Peffi : sf*Pstar)*oneOverRhoiRhoj;
      const auto Prhoj = (sameMatij ? Peffj : sf*Pstar)*oneOverRhoiRhoj;
      const auto deltaDvDt = (Prhoi*gradWi + Prhoj*gradWj) - Qacci - Qaccj;
      DvDti -= mj*deltaDvDt;
      DvDtj += mi*deltaDvDt;
      if (compatibleEnergyEvolution) pairAccelerations[kk] = -mj*deltaDvDt;  // Acceleration for i (j anti-symmetric)

      // Velocity gradient.
      const auto muij = vij.dot(rij);  
      auto yi =  1.0 + (Pstar-Peffi)/(Ki);
      auto yj =  1.0 + (Pstar-Peffj)/(Kj);
      if (muij > 0.0){
        const auto tempVar = yi;
        yi=yj;
        yj=tempVar;
      }
      const auto surfaceDecouplingi = (muij > 0.0 && Peffi < 0.0);
      const auto surfaceDecouplingj = (muij > 0.0 && Peffj < 0.0);
      auto kappai = (sameMatij ? 1.0 : max(0.0, min(2.0, 2.0*(Kj*voli*yi*gWj)/(Ki*volj*yj*gWi+Kj*voli*yi*gWj))));
      auto kappaj = (sameMatij ? 1.0 : max(0.0, min(2.0, 2.0-kappai)));
      kappai = (surfaceDecouplingi ? 0.0 : kappai);
      kappaj = (surfaceDecouplingj ? 0.0 : kappaj);

      auto deltaDvDxi = kappai*vij.dyad(gradWi);
      auto deltaDvDxj = kappaj*vij.dyad(gradWj);

      DvDxi -= volj*deltaDvDxi; 
      DvDxj -= voli*deltaDvDxj;

    // Specific thermal energy evolution.
      DepsDti += mj*(Prhoi*deltaDvDxi.Trace()-workQi);
      DepsDtj += mi*(Prhoj*deltaDvDxj.Trace()-workQj);

      // Linear gradient correction term.
      if(this->mCorrectVelocityGradient){
        Mi -= volj*rij.dyad(gradWi);
        Mj -= voli*rij.dyad(gradWj);
      }

      // Diffusion of density and specific thermal energy
      if (diffCoeff>tiny){
        const auto diffusion = diffCoeff*cij*etaij.dot(gradWij)/(etaMagij*etaMagij+tiny);
        const auto deltaRhoi = (sameMatij ? rhoj-rhoi : (Peffj-Peffi)/(max(ci*ci,0.0)+tiny) );
        const auto deltaRhoj = (sameMatij ? rhoi-rhoj : (Peffi-Peffj)/(max(cj*cj,0.0)+tiny) );
        const auto deltaEpsi = (sameMatij ? epsj-epsi : 0.0);
        const auto deltaEpsj = (sameMatij ? epsi-epsj : 0.0);
        DrhoDti -= volj*(deltaRhoi)*diffusion;
        DrhoDtj -= voli*(deltaRhoj)*diffusion;
        DepsDti -= mj*(deltaEpsi)*diffusion/rhoij;
        DepsDtj -= mi*(deltaEpsj)*diffusion/rhoij;
      }

      // Estimate of delta v (for XSPH).
      if (XSPH and (sameMatij)) {
        const auto wXSPHij = 0.5*(voli*Wi + volj*Wj);
        XSPHWeightSumi += wXSPHij;
        XSPHWeightSumj += wXSPHij;
        XSPHDeltaVi -= wXSPHij*vij;
        XSPHDeltaVj += wXSPHij*vij;
      }

    } // loop over pairs

    // Reduce the thread values to the master.
    threadReduceFieldLists<Dimension>(threadStack);

  }   // OpenMP parallel region
  TIME_SPHevalDerivs_pairs.stop();

  // Finish up the derivatives for each point.
  TIME_SPHevalDerivs_final.start();
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

      auto& rhoSumi = rhoSum(nodeListi, i);
      auto& normi = normalization(nodeListi, i);
      auto& DxDti = DxDt(nodeListi, i);
      auto& DrhoDti = DrhoDt(nodeListi, i);
      auto& DvDti = DvDt(nodeListi, i);
      auto& DepsDti = DepsDt(nodeListi, i);
      auto& DvDxi = DvDx(nodeListi, i);
      //auto& localDvDxi = localDvDx(nodeListi, i);
      auto& Mi = M(nodeListi, i);
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
      if (this->mCorrectVelocityGradient and
          std::abs(Mi.Determinant()) > 1.0e-10 and
          numNeighborsi > Dimension::pownu(2)) {
        Mi = Mi.Inverse();
        DvDxi = DvDxi*Mi;
      } 

      // Evaluate the continuity equation.
      DrhoDti -=  rhoi*DvDxi.Trace();

      // If needed finish the total energy derivative.
      if (evolveTotalEnergy) DepsDti = mi*(vi.dot(DvDti) + DepsDti);

      // Complete the moments of the node distribution for use in the ideal H calculation.
      weightedNeighborSumi = Dimension::rootnu(max(0.0, weightedNeighborSumi/Hdeti));
      massSecondMomenti /= Hdeti*Hdeti;

      // Determine the position evolution, based on whether we're doing XSPH or not.
      if (XSPH) {
        XSPHWeightSumi += Hdeti*mi/rhoi*W0;
        CHECK2(XSPHWeightSumi != 0.0, i << " " << XSPHWeightSumi);
        DxDti = vi + XSPHDeltaVi/max(tiny, XSPHWeightSumi);
      } else {
        DxDti = vi;
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
    }
  }
  TIME_SPHevalDerivs_final.stop();
  TIME_SPHevalDerivs.stop();
}

}