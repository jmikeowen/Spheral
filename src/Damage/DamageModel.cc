//---------------------------------Spheral++----------------------------------//
// DamageModel -- Base class for the damage physics models.
// This class just provides the basic interface for damage models, and does 
// not fill out the complete physics package interface.
//
// Created by JMO, Thu Sep 29 13:31:57 PDT 2005
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "DamageModel.hh"
#include "YoungsModulusPolicy.hh"
#include "LongitudinalSoundSpeedPolicy.hh"
#include "DamagedSoundSpeedPolicy.hh"
#include "Strength/SolidFieldNames.hh"
#include "NodeList/SolidNodeList.hh"
#include "DataBase/DataBase.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Field/FieldList.hh"
#include "Boundary/Boundary.hh"
#include "Kernel/TableKernel.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Utilities/NodeCoupling.hh"
#include "Utilities/DamagedNodeCoupling.hh"
#include "Utilities/DamageGradientNodeCoupling.hh"
#include "Utilities/ThreePointDamagedNodeCoupling.hh"
#include "Utilities/GeometricUtilities.hh"
#include "Utilities/safeInv.hh"
#include "Utilities/Timer.hh"

#include "boost/shared_ptr.hpp"

#include <string>
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
extern Timer TIME_Damage;
extern Timer TIME_DamageModel_finalize;

namespace Spheral {

//------------------------------------------------------------------------------
// Constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
DamageModel<Dimension>::
DamageModel(SolidNodeList<Dimension>& nodeList,
            const TableKernel<Dimension>& W,
            const double crackGrowthMultiplier,
            const DamageCouplingAlgorithm damageCouplingAlgorithm,
            const FlawStorageType& flaws):
  Physics<Dimension>(),
  mFlaws(SolidFieldNames::flaws, flaws),
  mNodeList(nodeList),
  mW(W),
  mCrackGrowthMultiplier(crackGrowthMultiplier),
  mCriticalNodesPerSmoothingScale(0.99),
  mDamageCouplingAlgorithm(damageCouplingAlgorithm),
  mYoungsModulus(SolidFieldNames::YoungsModulus, nodeList),
  mLongitudinalSoundSpeed(SolidFieldNames::longitudinalSoundSpeed, nodeList),
  mExcludeNode("Nodes excluded from damage", nodeList, 0),
  mNodeCouplingPtr(new NodeCoupling()),
  mComputeIntersectConnectivity(false),
  mRestart(registerWithRestart(*this)) {
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
DamageModel<Dimension>::
~DamageModel() {
}

//------------------------------------------------------------------------------
// Compute the generic Grady-Kipp (ala Benz-Asphaug) scalar damage time 
// derivative.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DamageModel<Dimension>::
computeScalarDDDt(const DataBase<Dimension>& /*dataBase*/,
                  const State<Dimension>& state,
                  const Scalar /*time*/,
                  const Scalar /*dt*/,
                  Field<Dimension, Scalar>& DDDt) const {

  // Pre-conditions.
  REQUIRE(DDDt.nodeListPtr() == &mNodeList);
  REQUIRE(mFlaws.nodeListPtr() == &mNodeList);

  // Get the state fields.
  const auto  clKey = State<Dimension>::buildFieldKey(SolidFieldNames::longitudinalSoundSpeed, mNodeList.name());
  const auto  HKey = State<Dimension>::buildFieldKey(HydroFieldNames::H, mNodeList.name());
  const auto& cl = state.field(clKey, 0.0);
  const auto& H = state.field(HKey, SymTensor::zero);

  // Constant multiplicative parameter for the crack growth.
  const auto A = mCrackGrowthMultiplier / mW.kernelExtent();

  // Iterate over the internal nodes.
  const auto ni = mNodeList.numInternalNodes();
#pragma omp parallel for
  for (auto i = 0u; i < ni; ++i) {
    if (mExcludeNode(i) == 1) {

      DDDt(i) = 0.0;

    } else {

      const double hrInverse = Dimension::rootnu(H(i).Determinant());
      DDDt(i) = A * cl(i) * hrInverse;

    }
    CHECK(DDDt(i) >= 0.0);
  }

  // Post-conditions.
  BEGIN_CONTRACT_SCOPE
  for (int i = 0; i != (int)mNodeList.numInternalNodes(); ++i) {
    if (mExcludeNode(i) == 1) {
      ENSURE(DDDt(i) == 0.0);
    }
  }
  END_CONTRACT_SCOPE
}

//------------------------------------------------------------------------------
// Register our state.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DamageModel<Dimension>::
registerState(DataBase<Dimension>& dataBase,
              State<Dimension>& state) {

  // Register Youngs modulus and the longitudinal sound speed.
  typedef typename State<Dimension>::PolicyPointer PolicyPointer;
  PolicyPointer EPolicy(new YoungsModulusPolicy<Dimension>());
  PolicyPointer clPolicy(new LongitudinalSoundSpeedPolicy<Dimension>());
  state.enroll(mYoungsModulus, EPolicy);
  state.enroll(mLongitudinalSoundSpeed, clPolicy);

  // Set the initial values for the Youngs modulus, sound speed, and pressure.
  typename StateDerivatives<Dimension>::PackageList dummyPackages;
  StateDerivatives<Dimension> derivs(dataBase, dummyPackages);
  EPolicy->update(state.key(mYoungsModulus), state, derivs, 1.0, 0.0, 0.0);
  clPolicy->update(state.key(mLongitudinalSoundSpeed), state, derivs, 1.0, 0.0, 0.0);
}

//------------------------------------------------------------------------------
// initialize (before evaluateDerivatives)
// This is where we update the damage coupling, which is stored in the
// connectivity pair list in f_couple.
// We can't do this during registerState because some coupling algorithms
// require ghost state to be updated first.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DamageModel<Dimension>::
initialize(const Scalar /*time*/,
           const Scalar /*dt*/,
           const DataBase<Dimension>& /*dataBase*/,
           State<Dimension>& state,
           StateDerivatives<Dimension>& /*derivs*/) {

  auto& connectivity = state.connectivityMap();
  auto& pairs = const_cast<NodePairList&>(connectivity.nodePairList());

  switch(mDamageCouplingAlgorithm) {
  case DamageCouplingAlgorithm::DirectDamage:
    break;

  case DamageCouplingAlgorithm::PairMaxDamage:
    mNodeCouplingPtr = std::make_shared<DamagedNodeCoupling<Dimension>>(state, pairs);
    break;

  case DamageCouplingAlgorithm::DamageGradient:
    mNodeCouplingPtr = std::make_shared<DamageGradientNodeCoupling<Dimension>>(state, mW, this->boundaryBegin(), this->boundaryEnd(), pairs);
    break;

  case DamageCouplingAlgorithm::ThreePointDamage:
    mNodeCouplingPtr = std::make_shared<ThreePointDamagedNodeCoupling<Dimension>>(state, mW, pairs);
    break;

  default:
    VERIFY2(false, "DamageModel ERROR: unhandled damage coupling algorithm case");
  }
  connectivity.coupling(mNodeCouplingPtr);
}

//------------------------------------------------------------------------------
// finalize (end of physics step)
//------------------------------------------------------------------------------
template<typename Dimension>
void
DamageModel<Dimension>::
finalize(const Scalar /*time*/, 
         const Scalar /*dt*/,
         DataBase<Dimension>& dataBase, 
         State<Dimension>& state,
         StateDerivatives<Dimension>& /*derivs*/) {
  TIME_Damage.start();
  TIME_DamageModel_finalize.start();

  // For 3pt damage, check if we should switch to using full intersection data
  // from the ConnectivityMap
  if (mDamageCouplingAlgorithm == DamageCouplingAlgorithm::ThreePointDamage) {
    const auto D = state.fields(SolidFieldNames::tensorDamage, SymTensor::zero);
    auto nD = 0u;
    const auto numNodeLists = D.numFields();
#pragma omp parallel
    {
      auto nD_thread = 0u;
      for (auto il = 0u; il < numNodeLists; ++il) {
        const auto n = D[il]->numInternalElements();
#pragma omp for
        for (auto i = 0u; i < n; ++i) {
          if (D(il,i).Trace() > 1.0e-3) ++nD_thread;
        }
      }
#pragma omp critical
      {
        nD += nD_thread;
      }
    }
    nD = allReduce(nD, MPI_SUM, Communicator::communicator());
    const auto ntot = std::max(1, dataBase.globalNumInternalNodes());
    const auto dfrac = double(nD)/double(ntot);
    mComputeIntersectConnectivity = (dfrac > 0.2);  // Should tune this number...
    if (Process::getRank() == 0) std::cout << "DamageModel dfrac = " << nD << "/" << ntot << " = " << dfrac << " : " << mComputeIntersectConnectivity << std::endl;
  }

  TIME_DamageModel_finalize.stop();
  TIME_Damage.stop();
}

//------------------------------------------------------------------------------
// Cull the flaws on each node to the single weakest one.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DamageModel<Dimension>::
cullToWeakestFlaws() {
#pragma omp parallel for
  for (auto i = 0u; i < mNodeList.numInternalNodes(); ++i) {
    auto& flaws = mFlaws[i];
    if (flaws.size() > 0) {
      const auto maxVal = *max_element(flaws.begin(), flaws.end());
      flaws = vector<double>(maxVal);
    }
  }
}

//------------------------------------------------------------------------------
// Access the set of nodes to be excluded from damage.
//------------------------------------------------------------------------------
template<typename Dimension>
vector<int>
DamageModel<Dimension>::
excludeNodes() const {
  vector<int> result;
  for (auto i = 0u; i != mNodeList.numInternalNodes(); ++i) {
    if (mExcludeNode(i) == 1.0) result.push_back(i);
  }
  return result;
}

template<typename Dimension>
void
DamageModel<Dimension>::
excludeNodes(vector<int> ids) {
  mExcludeNode = 0;
  for (vector<int>::const_iterator itr = ids.begin();
       itr != ids.end();
       ++itr) {
    REQUIRE(*itr >= 0 && *itr < (int)mNodeList.numInternalNodes());
    mExcludeNode(*itr) = 1;
  }
}

//------------------------------------------------------------------------------
// Compute a Field with the sum of the activation energies per node.
//------------------------------------------------------------------------------
template<typename Dimension>
Field<Dimension, typename Dimension::Scalar>
DamageModel<Dimension>::
sumActivationEnergiesPerNode() const {
  Field<Dimension, Scalar> result("Sum activation energies", mNodeList);
  for (auto i = 0u; i != mNodeList.numInternalNodes(); ++i) {
    const vector<double>& flaws = mFlaws(i);
    for (vector<double>::const_iterator flawItr = flaws.begin();
         flawItr != flaws.end();
         ++flawItr) {
      result(i) += *flawItr;
    }
  }
  return result;
}

//------------------------------------------------------------------------------
// Compute a Field with the number of flaws per node.
//------------------------------------------------------------------------------
template<typename Dimension>
Field<Dimension, typename Dimension::Scalar>
DamageModel<Dimension>::
numFlawsPerNode() const {
  Field<Dimension, Scalar> result("num flaws", mNodeList);
  for (auto i = 0u; i != mNodeList.numInternalNodes(); ++i) {
    result(i) = flawsForNode(i).size();
  }
  return result;
}

//------------------------------------------------------------------------------
// Dump the current state to the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DamageModel<Dimension>::
dumpState(FileIO& file, const string& pathName) const {
  file.write(mCrackGrowthMultiplier, pathName + "/crackGrowthMultiplier");
  file.write(mFlaws, pathName + "/flaws");
  file.write(mExcludeNode, pathName + "/excludeNode");
  file.write(mComputeIntersectConnectivity, pathName + "/computeIntersectConnectivity");
}

//------------------------------------------------------------------------------
// Restore the state from the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DamageModel<Dimension>::
restoreState(const FileIO& file, const string& pathName) {
  file.read(mCrackGrowthMultiplier, pathName + "/crackGrowthMultiplier");
  file.read(mFlaws, pathName + "/flaws");
  file.read(mExcludeNode, pathName + "/excludeNode");
  file.read(mComputeIntersectConnectivity, pathName + "/computeIntersectConnectivity");
}

}

