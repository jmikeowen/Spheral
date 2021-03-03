#include "Kernel/TableKernel.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Do we require ghost-ghost or intersection connectivity?
//------------------------------------------------------------------------------
template<typename Dimension>
inline
bool
DamageModel<Dimension>::
requireGhostConnectivity() const {
  return mDamageCouplingAlgorithm == DamageCouplingAlgorithm::ThreePointDamage;
}

template<typename Dimension>
inline
bool
DamageModel<Dimension>::
requireIntersectionConnectivity() const {
  return mComputeIntersectConnectivity;
}

//------------------------------------------------------------------------------
// The effective critical number of nodes per smoothing scale, below which we
// assume all flaws are active on a node.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
DamageModel<Dimension>::
criticalNodesPerSmoothingScale() const {
  return mCriticalNodesPerSmoothingScale;
}

template<typename Dimension>
inline
void
DamageModel<Dimension>::
criticalNodesPerSmoothingScale(double x) {
  VERIFY(x >= 0.0);
  mCriticalNodesPerSmoothingScale = x;
}

//------------------------------------------------------------------------------
// Return the set of flaw activation energies for the given node.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
const std::vector<double>
DamageModel<Dimension>::
flawsForNode(const size_t index) const {
  REQUIRE(index < mNodeList.numInternalNodes());
  REQUIRE(mFlaws.nodeListPtr() == &mNodeList);
  return mFlaws(index);
}

//------------------------------------------------------------------------------
// Access the internal parameters of the model.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
SolidNodeList<Dimension>&
DamageModel<Dimension>::
nodeList() {
  return mNodeList;
}

template<typename Dimension>
inline
const SolidNodeList<Dimension>&
DamageModel<Dimension>::
nodeList() const {
  return mNodeList;
}

template<typename Dimension>
inline
const TableKernel<Dimension>&
DamageModel<Dimension>::
kernel() const {
  return mW;
}

template<typename Dimension>
inline
double
DamageModel<Dimension>::
crackGrowthMultiplier() const {
  return mCrackGrowthMultiplier;
}

template<typename Dimension>
inline
DamageCouplingAlgorithm
DamageModel<Dimension>::
damageCouplingAlgorithm() const {
  return mDamageCouplingAlgorithm;
}

template<typename Dimension>
inline
const NodeCoupling&
DamageModel<Dimension>::
nodeCoupling() const {
  return *mNodeCouplingPtr;
}

//------------------------------------------------------------------------------
// Access the state fields.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
const Field<Dimension, typename Dimension::Scalar>&
DamageModel<Dimension>::
youngsModulus() const {
  return mYoungsModulus;
}

template<typename Dimension>
inline
const Field<Dimension, typename Dimension::Scalar>&
DamageModel<Dimension>::
longitudinalSoundSpeed() const {
  return mLongitudinalSoundSpeed;
}

//------------------------------------------------------------------------------
// Access the flaw activation strains.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
const typename DamageModel<Dimension>::FlawStorageType&
DamageModel<Dimension>::
flaws() const {
  return mFlaws;
}

template<typename Dimension>
inline
typename DamageModel<Dimension>::FlawStorageType&
DamageModel<Dimension>::
flaws() {
  return mFlaws;
}

}
