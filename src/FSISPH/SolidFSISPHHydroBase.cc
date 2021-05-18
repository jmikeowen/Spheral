//---------------------------------Spheral++----------------------------------//
// SolidSPHHydroBase -- SolidSPHHydro modified for large density discontinuities
//
// Created by JMO, Fri Jul 30 11:07:33 PDT 2010
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "Utilities/DamagedNodeCoupling.hh"
#include "SPH/SPHHydroBase.hh"
#include "SPH/SolidSPHHydroBase.hh"
#include "NodeList/SmoothingScaleBase.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Hydro/PressurePolicy.hh"
#include "Strength/SolidFieldNames.hh"
#include "NodeList/SolidNodeList.hh"
//#include "Strength/DeviatoricStressPolicy.hh"
//#include "Strength/BulkModulusPolicy.hh"
//#include "Strength/PlasticStrainPolicy.hh"
//#include "Strength/ShearModulusPolicy.hh"
//#include "Strength/YieldStrengthPolicy.hh"
//#include "Strength/StrengthSoundSpeedPolicy.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/IncrementFieldList.hh"
//#include "DataBase/IncrementBoundedState.hh"
//#include "DataBase/IncrementBoundedFieldList.hh"
//#include "DataBase/ReplaceFieldList.hh"
//#include "DataBase/ReplaceBoundedState.hh"
#include "DataBase/ReplaceBoundedFieldList.hh"
//#include "DataBase/CompositeFieldListPolicy.hh"
#include "ArtificialViscosity/ArtificialViscosity.hh"
#include "DataBase/DataBase.hh"
#include "Field/FieldList.hh"
#include "Field/NodeIterators.hh"
#include "Boundary/Boundary.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Utilities/timingUtilities.hh"
#include "Utilities/safeInv.hh"
#include "SolidMaterial/SolidEquationOfState.hh"
#include "Utilities/Timer.hh"

#include "FSISPH/FSISpecificThermalEnergyPolicy.hh"
#include "FSISPH/SolidFSISPHHydroBase.hh"
#include "FSISPH/computeSurfaceNormals.hh"
#include "FSISPH/computeFSISPHSumMassDensity.hh"

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

extern Timer TIME_SolidFSISPHpreStepInitialize;
extern Timer TIME_SolidFSISPHinitialize;
extern Timer TIME_SolidFSISPHregisterDerivs;
extern Timer TIME_SolidFSISPHregisterState;

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
SolidFSISPHHydroBase<Dimension>::
SolidFSISPHHydroBase(const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
                  DataBase<Dimension>& dataBase,
                  ArtificialViscosity<Dimension>& Q,
                  const TableKernel<Dimension>& W,
                  const double filter,
                  const double cfl,
                  const double surfaceForceCoefficient,
                  const double densityStabilizationCoefficient,
                  const double densityDiffusionCoefficient,
                  const double specificThermalEnergyDiffusionCoefficient,
                  const std::vector<int> sumDensityNodeLists,
                  const bool useVelocityMagnitudeForDt,
                  const bool compatibleEnergyEvolution,
                  const bool evolveTotalEnergy,
                  const bool gradhCorrection,
                  const bool XSPH,
                  const bool correctVelocityGradient,
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
                               W,  //WPi
                               W,  //WGrad
                               filter,
                               cfl,
                               useVelocityMagnitudeForDt,
                               compatibleEnergyEvolution,
                               evolveTotalEnergy,
                               gradhCorrection, 
                               XSPH,
                               correctVelocityGradient,
                               true, // sumMassDensityOverAllNodeLists
                               densityUpdate,
                               HUpdate,
                               epsTensile,
                               nTensile,
                               damageRelieveRubble,
                               negativePressureInDamage,
                               strengthInDamage,
                               xmin,
                               xmax),
  mSurfaceForceCoefficient(surfaceForceCoefficient),
  mDensityStabilizationCoefficient(densityStabilizationCoefficient),
  mDensityDiffusionCoefficient(densityDiffusionCoefficient),
  mSpecificThermalEnergyDiffusionCoefficient(specificThermalEnergyDiffusionCoefficient),
  mApplySelectDensitySum(false),
  mSumDensityNodeLists(sumDensityNodeLists),
  mSurfaceNormals(FieldStorageType::CopyFields){
     mSurfaceNormals = dataBase.newSolidFieldList(Vector::zero,  "FSISurfaceNormals");

    // see if we're summing density for any nodelist
    auto numNodeLists = dataBase.numNodeLists();
    for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
      if (sumDensityNodeLists[nodeListi]==1){
        mApplySelectDensitySum = true;
      } 
    }
    
  }

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
SolidFSISPHHydroBase<Dimension>::
~SolidFSISPHHydroBase() {
}


//------------------------------------------------------------------------------
// Register states
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidFSISPHHydroBase<Dimension>::
registerState(DataBase<Dimension>& dataBase,
              State<Dimension>& state) {
  TIME_SolidFSISPHregisterState.start();

  SolidSPHHydroBase<Dimension>::registerState(dataBase,state);
  
  // Override the specific thermal energy policy
  typedef typename State<Dimension>::PolicyPointer PolicyPointer;
  FieldList<Dimension, Scalar> eps = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  CHECK(eps.numFields() == dataBase.numFluidNodeLists());
  PolicyPointer epsPolicy(new FSISpecificThermalEnergyPolicy<Dimension>(dataBase));
  state.enroll(eps, epsPolicy);
  

  TIME_SolidFSISPHregisterState.stop();
}

//------------------------------------------------------------------------------
// Register Derivs
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidFSISPHHydroBase<Dimension>::
registerDerivatives(DataBase<Dimension>&  dataBase,
                    StateDerivatives<Dimension>& derivs) {
  TIME_SolidFSISPHregisterDerivs.start();

  // Call the ancestor method.
  SolidSPHHydroBase<Dimension>::registerDerivatives(dataBase, derivs);


  TIME_SolidFSISPHregisterDerivs.stop();
}
//------------------------------------------------------------------------------
// FSI specialized density summmation
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidFSISPHHydroBase<Dimension>::
preStepInitialize(const DataBase<Dimension>& dataBase, 
                  State<Dimension>& state,
                  StateDerivatives<Dimension>& /*derivs*/) {
  TIME_SolidFSISPHpreStepInitialize.start();
  if (mApplySelectDensitySum){
      const auto& connectivityMap = dataBase.connectivityMap();
      const auto& position = state.fields(HydroFieldNames::position, Vector::zero);
      const auto& mass = state.fields(HydroFieldNames::mass, 0.0);
      const auto& H = state.fields(HydroFieldNames::H, SymTensor::zero);
      const auto& W = this->kernel();
            auto  massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
      computeFSISPHSumMassDensity(connectivityMap, W, mSumDensityNodeLists, position, mass, H, massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->applyFieldListGhostBoundary(massDensity);
      for (auto boundaryItr = this->boundaryBegin(); boundaryItr < this->boundaryEnd(); ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
  }
  TIME_SolidFSISPHpreStepInitialize.stop();
}



//------------------------------------------------------------------------------
// FSI specialized of the initialize method
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidFSISPHHydroBase<Dimension>::
initialize(const typename Dimension::Scalar time,
           const typename Dimension::Scalar dt,
           const DataBase<Dimension>& dataBase,
           State<Dimension>& state,
           StateDerivatives<Dimension>& derivs) {
  TIME_SolidFSISPHinitialize.start();

  const TableKernel<Dimension>& W = this->kernel();
  ArtificialViscosity<Dimension>& Q = this->artificialViscosity();
  Q.initialize(dataBase, 
               state,
               derivs,
               this->boundaryBegin(),
               this->boundaryEnd(),
               time, 
               dt,
               W);

  // put some trigger for this in Slip interface bc?
  if (true){  
    const auto& connectivityMap = dataBase.connectivityMap();
    const auto& position = state.fields(HydroFieldNames::position, Vector::zero);
    const auto& mass = state.fields(HydroFieldNames::mass, 0.0);
    const auto& massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
    const auto& H = state.fields(HydroFieldNames::H, SymTensor::zero);
          auto& normals = this->mSurfaceNormals;
    normals.Zero();
    computeSurfaceNormals(connectivityMap,
                          W,
                          position,
                          mass,
                          massDensity,
                          H,
                          normals);


    for (ConstBoundaryIterator boundaryItr = this->boundaryBegin();
         boundaryItr != this->boundaryEnd();
         ++boundaryItr) {
        (*boundaryItr)->applyFieldListGhostBoundary(normals);
    }
    for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
           boundaryItr != this->boundaryEnd();
           ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
  }
  // We depend on the caller knowing to finalize the ghost boundaries!
  TIME_SolidFSISPHinitialize.stop();
}

//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SolidFSISPHHydroBase<Dimension>::
evaluateDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar dt,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivatives) const {


  // Get the ArtificialViscosity.
  auto& Q = this->artificialViscosity();

  // The kernels and such.
  const auto& W = this->kernel();
  const auto& smoothingScaleMethod = this->smoothingScaleMethod();

  // A few useful constants we'll use in the following loop.
  const auto tiny = 1.0e-30;
  const auto W0 = W(0.0, 1.0);
  const auto epsTensile = this->epsilonTensile();
  //const auto compatibleEnergy = this->compatibleEnergyEvolution();
  const auto damageRelieveRubble = this->damageRelieveRubble();
  const auto rhoDiffusionCoeff = this->densityDiffusionCoefficient();
  const auto epsDiffusionCoeff = this->specificThermalEnergyDiffusionCoefficient();
  const auto rhoStabilizeCoeff = this->densityStabilizationCoefficient();
  const auto surfaceForceCoeff = this->surfaceForceCoefficient();
  const auto XSPH = this->XSPH();

  // the surface normals 
  const auto& interfaceNormals = this -> mSurfaceNormals;

  // The connectivity.
  const auto& connectivityMap = dataBase.connectivityMap();
  const auto& nodeLists = connectivityMap.nodeLists();
  const auto numNodeLists = nodeLists.size();

  // Get the state and derivative FieldLists.
  const auto mass = state.fields(HydroFieldNames::mass, 0.0);
  const auto position = state.fields(HydroFieldNames::position, Vector::zero);
  const auto velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  const auto massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  const auto specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  const auto H = state.fields(HydroFieldNames::H, SymTensor::zero);
  const auto pressure = state.fields(HydroFieldNames::pressure, 0.0);
  const auto soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);
  //auto omega = state.fields(HydroFieldNames::omegaGradh, 0.0);
  const auto S = state.fields(SolidFieldNames::deviatoricStress, SymTensor::zero);
  const auto mu = state.fields(SolidFieldNames::shearModulus, 0.0);
  const auto damage = state.fields(SolidFieldNames::tensorDamage, SymTensor::zero);
  //const auto gradDamage = state.fields(SolidFieldNames::damageGradient, Vector::zero);
  const auto fragIDs = state.fields(SolidFieldNames::fragmentIDs, int(1));
  const auto pTypes = state.fields(SolidFieldNames::particleTypes, int(0));
  //const auto K = state.fields(SolidFieldNames::bulkModulus, 0.0);
  
  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(velocity.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(specificThermalEnergy.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);
  CHECK(pressure.size() == numNodeLists);
  CHECK(soundSpeed.size() == numNodeLists);
  //CHECK(omega.size() == numNodeLists);
  CHECK(S.size() == numNodeLists);
  CHECK(mu.size() == numNodeLists);
  CHECK(damage.size() == numNodeLists);
  //CHECK(gradDamage.size() == numNodeLists);
  CHECK(fragIDs.size() == numNodeLists);
  CHECK(pTypes.size() == numNodeLists);
  //CHECK(K.size() == numNodeLists);

  // Derivative FieldLists.
  //auto  rhoSum = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
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
  //auto  maxViscousPressure = derivatives.fields(HydroFieldNames::maxViscousPressure, 0.0);
  //auto  effViscousPressure = derivatives.fields(HydroFieldNames::effectiveViscousPressure, 0.0);
  //auto  rhoSumCorrection = derivatives.fields(HydroFieldNames::massDensityCorrection, 0.0);
  //auto  viscousWork = derivatives.fields(HydroFieldNames::viscousWork, 0.0);
  auto& pairAccelerations = derivatives.getAny(HydroFieldNames::pairAccelerations, vector<Vector>());
  auto  XSPHWeightSum = derivatives.fields(HydroFieldNames::XSPHWeightSum, 0.0);
  auto  XSPHDeltaV = derivatives.fields(HydroFieldNames::XSPHDeltaV, Vector::zero);
  auto  weightedNeighborSum = derivatives.fields(HydroFieldNames::weightedNeighborSum, 0.0);
  auto  massSecondMoment = derivatives.fields(HydroFieldNames::massSecondMoment, SymTensor::zero);
  auto  DSDt = derivatives.fields(IncrementFieldList<Dimension, SymTensor>::prefix() + SolidFieldNames::deviatoricStress, SymTensor::zero);
  
  //CHECK(rhoSum.size() == numNodeLists);
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
  //CHECK(maxViscousPressure.size() == numNodeLists);
  //CHECK(effViscousPressure.size() == numNodeLists);
  //CHECK(rhoSumCorrection.size() == numNodeLists);
  //CHECK(viscousWork.size() == numNodeLists);
  CHECK(XSPHWeightSum.size() == numNodeLists);
  CHECK(XSPHDeltaV.size() == numNodeLists);
  CHECK(weightedNeighborSum.size() == numNodeLists);
  CHECK(massSecondMoment.size() == numNodeLists);
  CHECK(DSDt.size() == numNodeLists);

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();

  // Size up the pair-wise accelerations before we start.
  pairAccelerations.resize(2*npairs);

  // The scale for the tensile correction.
  const auto& nodeList = mass[0]->nodeList();
  const auto  nPerh = nodeList.nodesPerSmoothingScale();
  const auto  WnPerh = W(1.0/nPerh, 1.0);




//M corr needs to be calculated beforehand 
//to be consistently applied to the acceleration
//and the time derivative of internal energy
if(this->correctVelocityGradient()){


#pragma omp parallel
  {
    // Thread private  scratch variables.
    int i, j, nodeListi, nodeListj;
    //Scalar Wi, gWi, Wj, gWj;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;
    auto M_thread = M.threadCopy(threadStack);

#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      //const auto start = Timing::currentTime();
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
            auto  Hdeti = Hi.Determinant();
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      // Get the state for node j
      const auto& rj = position(nodeListj, j);
      const auto& mj = mass(nodeListj, j);
      const auto& rhoj = massDensity(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
            auto  Hdetj = Hj.Determinant();
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& Mi = M_thread(nodeListi,i);
      auto& Mj = M_thread(nodeListj,j);


      // Kernels
      //--------------------------------------
      const auto rij = ri - rj;
      //const auto Hij = 0.5*(Hi+Hj);
      //const auto etaij = Hij*rij;
      //const auto Hdetij = Hij.Determinant();

      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);

      // Symmetrized kernel weight and gradient.
      const auto gWi = W.gradValue(etaMagi, Hdeti);
      const auto gWj = W.gradValue(etaMagj, Hdetj);
      const auto Hetai = Hi*etai.unitVector();
      const auto Hetaj = Hj*etaj.unitVector();
      auto gradWi = gWi*Hetai;
      auto gradWj = gWj*Hetaj;

     // averaged things.
      //const auto Wij = 0.5*(Wi+Wj); 
     // const auto gWij = 0.5*(gWi+gWj);
      

      //Wi & Wj --> Wij for interface better agreement DrhoDt and DepsDt
      if (!(nodeListi==nodeListj)){
        const auto gradWij = 0.5*(gradWi+gradWj);
        //Hdeti =  1.0*Hdetij;
        //Hdetj = 1.0*Hdetij;
       // Wi = 1.0*Wij;
        //Wj = 1.0*Wij;
        //gWi = 1.0*gWij;
        //gWj = 1.0*gWij;
       gradWi = 1.0*gradWij;
       gradWj = 1.0*gradWij;
      }

      // linear velocity gradient correction
      //---------------------------------------------------------------
      Mi -=  mj/rhoj * rij.dyad(gradWi);
      Mj -=  mi/rhoi * rij.dyad(gradWj);

    } // loop over pairs
      // Reduce the thread values to the master.
    threadReduceFieldLists<Dimension>(threadStack);
  }   // OpenMP parallel region

   
    for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
      const auto& nodeList = mass[nodeListi]->nodeList();
      const auto ni = nodeList.numInternalNodes();
#pragma omp parallel for
      for (auto i = 0u; i < ni; ++i) {

        const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
        auto& Mi = M(nodeListi, i);

        const auto goodM = std::abs(Mi.Determinant()) > 1.0e-10 and numNeighborsi > Dimension::pownu(2);
        Mi =  (goodM? Mi.Inverse(): Tensor::one);
      } 
    }


  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin();
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
      (*boundaryItr)->applyFieldListGhostBoundary(M);
    }
  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();

} // if correct velocity gradient



// Now we calculate  the hydro deriviatives
// Walk all the interacting pairs.
#pragma omp parallel
  {
    // Thread private  scratch variables.
    int i, j, nodeListi, nodeListj;
    Scalar Wi, gWi, Wj, gWj;
    Tensor QPiij, QPiji;
    SymTensor sigmai, sigmaj;
    Vector sigmarhoi, sigmarhoj;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;
    auto DvDt_thread = DvDt.threadCopy(threadStack);
    auto DepsDt_thread = DepsDt.threadCopy(threadStack);
    auto DrhoDt_thread = DrhoDt.threadCopy(threadStack);
    auto DSDt_thread = DSDt.threadCopy(threadStack);
    auto DvDx_thread = DvDx.threadCopy(threadStack);
    auto localDvDx_thread = localDvDx.threadCopy(threadStack);
    auto localM_thread = localM.threadCopy(threadStack);
    auto XSPHWeightSum_thread = XSPHWeightSum.threadCopy(threadStack);
    auto XSPHDeltaV_thread = XSPHDeltaV.threadCopy(threadStack);
    auto weightedNeighborSum_thread = weightedNeighborSum.threadCopy(threadStack);
    auto massSecondMoment_thread = massSecondMoment.threadCopy(threadStack);
    //auto maxViscousPressure_thread = maxViscousPressure.threadCopy(threadStack, ThreadReduction::MAX);
    //auto effViscousPressure_thread = effViscousPressure.threadCopy(threadStack);
    //auto viscousWork_thread = viscousWork.threadCopy(threadStack);

#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      //const auto start = Timing::currentTime();
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& epsi = specificThermalEnergy(nodeListi,i);
      const auto& Pi = pressure(nodeListi, i);
      const auto& ci = soundSpeed(nodeListi, i);
      const auto& Si = S(nodeListi, i);
      const auto& normi = interfaceNormals(nodeListi,i);
      const auto& pTypei = pTypes(nodeListi, i);
      const auto  voli = mi/rhoi;
      const auto  mui = max(mu(nodeListi,i),tiny);
      const auto  Ki = max(tiny,rhoi*ci*ci)+4.0/3.0*mui;
      auto  Hdeti = Hi.Determinant();
       //const auto fragIDi = fragIDs(nodeListi, i);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& DvDti = DvDt_thread(nodeListi, i);
      auto& DrhoDti = DrhoDt_thread(nodeListi, i);
      auto& DepsDti = DepsDt_thread(nodeListi, i);
      auto& DSDti = DSDt_thread(nodeListi, i);
      auto& DvDxi = DvDx_thread(nodeListi, i);
      auto& localDvDxi = localDvDx_thread(nodeListi, i);
      const auto& Mi = M(nodeListi, i);
      auto& localMi = localM_thread(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum_thread(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV_thread(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum_thread(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment_thread(nodeListi, i);

      // Get the state for node j
      const auto& rj = position(nodeListj, j);
      const auto& vj = velocity(nodeListj, j);
      const auto& mj = mass(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto& rhoj = massDensity(nodeListj, j);
      const auto& epsj = specificThermalEnergy(nodeListj,j);
      const auto& Pj = pressure(nodeListj, j);
      const auto& cj = soundSpeed(nodeListj, j);
      const auto& Sj = S(nodeListj, j);
      const auto& pTypej = pTypes(nodeListj, j);
      const auto& normj = interfaceNormals(nodeListj,j);
      const auto  volj = mj/rhoj;
      const auto  muj = max(mu(nodeListj,j),tiny);
      const auto  Kj = max(tiny,rhoj*cj*cj)+4.0/3.0*muj;
            auto  Hdetj = Hj.Determinant();
      //const auto fragIDj = fragIDs(nodeListj, j);
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& DvDtj = DvDt_thread(nodeListj, j);
      auto& DrhoDtj = DrhoDt_thread(nodeListj, j);
      auto& DepsDtj = DepsDt_thread(nodeListj, j);
      auto& DSDtj = DSDt_thread(nodeListj, j);
      auto& DvDxj = DvDx_thread(nodeListj, j);
      auto& localDvDxj = localDvDx_thread(nodeListj, j);
      const auto& Mj = M(nodeListj,j);
      auto& localMj = localM_thread(nodeListj, j);
      auto& XSPHWeightSumj = XSPHWeightSum_thread(nodeListj, j);
      auto& XSPHDeltaVj = XSPHDeltaV_thread(nodeListj, j);
      auto& weightedNeighborSumj = weightedNeighborSum_thread(nodeListj, j);
      auto& massSecondMomentj = massSecondMoment_thread(nodeListj, j);

      // Flag if this is a contiguous material pair or not.
      const auto sameMatij =  (nodeListi == nodeListj);// and fragIDi == fragIDj); 

      // Flag if at least one particle is free (0).
      const auto freeParticle = (pTypei == 0 or pTypej == 0);

      const auto fDij = pairs[kk].f_couple;

      // Kernels
      //--------------------------------------
      const auto rij = ri - rj;
      const auto Hij = 0.5*(Hi+Hj);
      const auto etaij = Hij*rij;
      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagij = etaij.magnitude();
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagij >= 0.0);
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);

      // Symmetrized kernel weight and gradient.
      std::tie(Wi, gWi) = W.kernelAndGradValue(etaMagi, Hdeti);
      std::tie(Wj, gWj) = W.kernelAndGradValue(etaMagj, Hdetj);
      const auto Hetai = Hi*etai.unitVector();
      const auto Hetaj = Hj*etaj.unitVector();
      auto gradWi = gWi*Hetai;
      auto gradWj = gWj*Hetaj;
      const auto gradWij = 0.5*(gradWi+gradWj);

      // Wi & Wj --> Wij for interface better agreement DrhoDt and DepsDt
      if (!sameMatij){
        const auto Hdetij = Hij.Determinant();
        const auto Wij = 0.5*(Wi+Wj); 
        const auto gWij = 0.5*(gWi+gWj);

        Hdeti =  1.0*Hdetij;
        Hdetj = 1.0*Hdetij;
        Wi = 1.0*Wij;
        Wj = 1.0*Wij;
        gWi = 1.0*gWij;
        gWj = 1.0*gWij;
        gradWi = 1.0*gradWij;
        gradWj = 1.0*gradWij;
      }

      //frame as a kernel correction
      if(this->correctVelocityGradient()){
        gradWi = Mi.Transpose()*gradWi;
        gradWj = Mj.Transpose()*gradWj;
      }

      // Zero'th and second moment of the node distribution -- used for the
      // ideal H calculation.
      //---------------------------------------------------------------
      const auto rij2 = rij.magnitude2();
      const auto thpt = rij.selfdyad()*safeInvVar(rij2*rij2*rij2);
      weightedNeighborSumi += abs(gWi);
      weightedNeighborSumj += abs(gWj);
      massSecondMomenti += gradWi.magnitude2()*thpt;
      massSecondMomentj += gradWj.magnitude2()*thpt;

      // Stress state
      //---------------------------------------------------------------
      const auto rhoij = 0.5*(rhoi+rhoj); 
      const auto cij = 0.5*(ci+cj); 
      const auto vij = vi - vj;

      std::tie(QPiij, QPiji) = Q.Piij(nodeListi, i, nodeListj, j,
                                      ri, etaij, vi, rhoij, cij, Hij,  
                                      rj, etaij, vj, rhoij, cij, Hij); 

      // stresses for interacting pairs.
      if (sameMatij) {
        sigmai = fDij*Si - Pi * SymTensor::one;
        sigmaj = fDij*Sj - Pj * SymTensor::one;
      }else {
        //const auto PSi = rij.dot(Si.dot(rij))/rij2;
        //const auto PSj = rij.dot(Sj.dot(rij))/rij2;
        //const auto Pstar = ((Pi-PSi)*rhoj+(Pj-PSj)*rhoi)/(rhoi+rhoj);
        // if slip 
        const auto avInterfaceReduceri =  abs((normi).unitVector().dot(vij.unitVector()));
        const auto avInterfaceReducerj =  abs((normj).unitVector().dot(vij.unitVector()));
        QPiij *= avInterfaceReduceri*avInterfaceReducerj;
        QPiji *= avInterfaceReduceri*avInterfaceReducerj;

        const auto Peffi = max(Pi,0.0);
        const auto Peffj = max(Pj,0.0);
        const auto Pstar = ((Peffi)*rhoj+(Peffj)*rhoi)/(rhoi+rhoj);
        sigmai = -Pstar*SymTensor::one;
        sigmaj = -Pstar*SymTensor::one;
      }
      
      // Compute the tensile correction to add to the stress as described in 
      // Gray, Monaghan, & Swift (Comput. Methods Appl. Mech. Eng., 190, 2001)
      const auto fi = epsTensile*FastMath::pow4(Wi/(Hdeti*WnPerh));
      const auto fj = epsTensile*FastMath::pow4(Wj/(Hdetj*WnPerh));
      const auto Ri = fi*tensileStressCorrection(sigmai);
      const auto Rj = fj*tensileStressCorrection(sigmaj);
      sigmai += Ri;
      sigmaj += Rj;

      // accelerations
      //---------------------------------------------------------------
      const auto rhoirhoj = 1.0/(rhoi*rhoj);
      const auto sf = (sameMatij ? 1.0 : 1.0 + surfaceForceCoeff*abs((rhoi-rhoj)/(rhoi+rhoj+tiny)));
      
      sigmarhoi = sf*((rhoirhoj*sigmai-0.5*QPiij))*gradWi;
      sigmarhoj = sf*((rhoirhoj*sigmaj-0.5*QPiji))*gradWj;
      
      const auto deltaDvDt = sigmarhoi+sigmarhoj;
      pairAccelerations[2*kk+1] = deltaDvDt;

      if (freeParticle) {
        DvDti += mj*deltaDvDt;
        DvDtj -= mi*deltaDvDt;
      } 
      
      // construct our interface velocity
      //-----------------------------------------------------------
      // deconstruct into parallel and perpendicular
      const auto rhatij = rij.unitVector();
      const auto ui = vi.dot(rhatij);
      const auto uj = vj.dot(rhatij);
      const auto wi = vi - ui*rhatij;
      const auto wj = vj - uj*rhatij;
      const auto umin = min(ui,uj);
      const auto umax = max(ui,uj);

      // material property avg weights
      const auto Ci = Ki*volj*gWi;
      const auto Cj = Kj*voli*gWj;
      const auto weightUi = max(0.0, min(1.0, Ci/(Ci+Cj)));
      const auto weightUj = max(0.0, min(1.0, 1.0 - weightUi));
      const auto weightWi = max(0.0, min(1.0, mui/(mui+muj)));
      const auto weightWj = max(0.0, min(1.0, 1.0 - weightWi));

      // same mat no damage defaults to avg
      Scalar ustar = 0.5*(ui+uj);
      Vector wstar = 0.5*(wi+wj);

      // diff materials use mat properties to weight things
      // for damaged/undamaged treat as two diff materials
      if(!sameMatij){
        ustar = weightUi*ui + weightUj*uj;
        wstar = weightWi*wi + weightWj*wj; 
      }
      if(sameMatij and fDij<0.999){ 
        const auto ustarDamaged = weightUi*ui + weightUj*uj;
        const auto wstarDamaged = weightWi*wi + weightWj*wj;
        wstar = fDij*wstar + (1.0-fDij)*wstarDamaged;
        ustar = fDij*ustar + (1.0-fDij)*ustarDamaged;
      }

      // additional stabilization 
      if (rhoStabilizeCoeff>tiny){
          const auto denom = safeInv(max(tiny,(sameMatij      ?
                                               max(rhoi,rhoj) :
                                               max(rhoi*ci*ci,rhoj*cj*cj))));

          const auto ustarStabilizer =  (sameMatij  ?
                                        (rhoj-rhoi) :
                                        (Pj-Pi)     )*denom;

          ustar += rhoStabilizeCoeff * min(0.1, max(-0.1, ustarStabilizer)) * cij*etaMagij;
      }

      //bound it 
      ustar = min(max(ustar,umin),umax);

      // reassemble 
      const auto vstar = ustar*rhatij + wstar;

      // energy conservation
      // ----------------------------------------------------------
      DepsDti -= 2.0*mj*sigmarhoi.dot(vi-vstar);
      DepsDtj -= 2.0*mi*sigmarhoj.dot(vstar-vj);
      pairAccelerations[2*kk][0] = -2.0*mj*sigmarhoi.dot(vi-vstar); 
      pairAccelerations[2*kk][1] = -2.0*mi*sigmarhoj.dot(vstar-vj);

      // velocity gradient --> continuity
      //-----------------------------------------------------------
      auto deltaDvDxi = 2.0*(vi-vstar).dyad(gradWi);
      auto deltaDvDxj = 2.0*(vstar-vj).dyad(gradWj);

      DvDxi -= volj*deltaDvDxi;
      DvDxj -= voli*deltaDvDxj;

      // intact material
      if (sameMatij) {
        localMi -= fDij*volj*rij.dyad(gradWi);
        localMj -= fDij*voli*rij.dyad(gradWj);
        localDvDxi -= fDij*volj*(deltaDvDxi);
        localDvDxj -= fDij*voli*(deltaDvDxj); 
      }

      // diffusions
      //-----------------------------------------------------------
      if (sameMatij and rhoDiffusionCoeff>tiny){
        const auto diffusion =  rhoDiffusionCoeff*(rhoi-rhoj)*cij*etaij.dot(gradWij)/(etaMagij*etaMagij+tiny);
        DrhoDti += volj*diffusion;
        DrhoDtj -= voli*diffusion;
      }

      if (sameMatij and epsDiffusionCoeff>tiny){
       const auto diffusion =  epsDiffusionCoeff*(epsi-epsj)*cij*etaij.dot(gradWij)/(rhoij*etaMagij*etaMagij+tiny);
       DepsDti += mj*diffusion;
       DepsDtj -= mi*diffusion;
       pairAccelerations[2*kk][0] += mj*diffusion; 
       pairAccelerations[2*kk][1] -= mi*diffusion;
      }

      // rigorous enforcement of single-valued stress-state at interface
      if (!sameMatij){
        const auto diffusion = (Si-Sj)*cij*etaij.dot(gradWij)/(etaMagij*etaMagij+tiny);
        DSDti += volj*diffusion;
        DSDtj -= voli*diffusion;
      }

      // XSPH
      //-----------------------------------------------------------
      if (XSPH and sameMatij) {
        XSPHWeightSumi += volj*Wi;
        XSPHWeightSumj += voli*Wj;
        XSPHDeltaVi -= volj*Wi*vij;
        XSPHDeltaVj += voli*Wj*vij;
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
      const auto& Si = S(nodeListi, i);
      const auto& mui = mu(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& DxDti = DxDt(nodeListi, i);
      auto& DrhoDti = DrhoDt(nodeListi, i);
      auto& DvDxi = DvDx(nodeListi, i);
      auto& localDvDxi = localDvDx(nodeListi, i);
     // auto& Mi = M(nodeListi, i);
      auto& localMi = localM(nodeListi, i);
      auto& DHDti = DHDt(nodeListi, i);
      auto& Hideali = Hideal(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment(nodeListi, i);
      auto& DSDti = DSDt(nodeListi, i);
      
      // Complete the moments of the node distribution for use in the ideal H calculation.
      weightedNeighborSumi = Dimension::rootnu(max(0.0, weightedNeighborSumi/Hdeti));
      massSecondMomenti /= Hdeti*Hdeti;

      DrhoDti -=  rhoi*DvDxi.Trace();

      DxDti = vi;
       if (XSPH) {
        CHECK(XSPHWeightSumi >= 0.0);
        XSPHWeightSumi += Hdeti*mi/rhoi*W0 + tiny;
        DxDti += 0.25*XSPHDeltaVi/XSPHWeightSumi;
      }

    
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

      const auto Di = (damageRelieveRubble ? 
                       max(0.0, min(1.0, damage(nodeListi, i).Trace() - 1.0)) :
                       0.0);

      if (std::abs(localMi.Determinant()) > 1.0e-10 and
        numNeighborsi > Dimension::pownu(2)) {
        localMi = localMi.Inverse();
        localDvDxi = localDvDxi*localMi;
      }

      // Determine the deviatoric stress evolution.
      const auto deformation = localDvDxi.Symmetric();
      const auto spin = localDvDxi.SkewSymmetric();
      const auto deviatoricDeformation = deformation - (deformation.Trace()/3.0)*SymTensor::one;
      const auto spinCorrection = (spin*Si + Si*spin).Symmetric();
      DSDti += spinCorrection + (2.0*mui)*deviatoricDeformation;

      // In the presence of damage, add a term to reduce the stress on this point.
      DSDti = (1.0 - Di)*DSDti - 0.25/dt*Di*Si;
    }
  }
}

}



