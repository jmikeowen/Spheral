//---------------------------------Spheral++----------------------------------//
// DataBase -- The central point to store NodeLists and fields, to serve the 
//             Spheral++ physics modules.
//
// Created by JMO, Sun Feb  6 13:44:49 PST 2000
//----------------------------------------------------------------------------//
#include "DataBase.hh"
#include "Geometry/Dimension.hh"
#include "Field/NodeIterators.hh"
#include "Field/Field.hh"
#include "Field/FieldList.hh"
#include "Kernel/TableKernel.hh"
#include "Material/EquationOfState.hh"
#include "Utilities/testBoxIntersection.hh"
#include "Utilities/safeInv.hh"
#include "State.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Utilities/globalBoundingVolumes.hh"
#include "Utilities/allReduce.hh"
#include "Distributed/Communicator.hh"
#include "Utilities/DBC.hh"

#ifdef USE_MPI
extern "C" {
#include "mpi.h"
}
#endif

#include <algorithm>
#include <memory>
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::abs;
using std::shared_ptr;

namespace Spheral {

//------------------------------------------------------------------------------
// Empty constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
DataBase<Dimension>::DataBase():
  mNodeListPtrs(0),
  mFluidNodeListPtrs(0),
  mFluidNodeListAsNodeListPtrs(0),
  mSolidNodeListPtrs(0),
  mSolidNodeListAsNodeListPtrs(0),
  mConnectivityMapPtr(new ConnectivityMap<Dimension>()) {
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
DataBase<Dimension>::~DataBase() {
}

//------------------------------------------------------------------------------
// Assignment.
//------------------------------------------------------------------------------
template<typename Dimension>
DataBase<Dimension>&
DataBase<Dimension>::
operator=(const DataBase<Dimension>& rhs) {
  REQUIRE(rhs.valid());
  if (this != &rhs) {
    mNodeListPtrs = rhs.mNodeListPtrs;
    mFluidNodeListPtrs = rhs.mFluidNodeListPtrs;
    mFluidNodeListAsNodeListPtrs = rhs.mFluidNodeListAsNodeListPtrs;
    mSolidNodeListPtrs = rhs.mSolidNodeListPtrs;
    mSolidNodeListAsNodeListPtrs = rhs.mSolidNodeListAsNodeListPtrs;
    mConnectivityMapPtr = std::shared_ptr<ConnectivityMap<Dimension> >(new ConnectivityMap<Dimension>());
  }
  ENSURE(valid());
  return *this;
}

//------------------------------------------------------------------------------
// Global numbers of nodes in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
int
DataBase<Dimension>::globalNumInternalNodes() const {
  int localResult = numInternalNodes();
  int result = localResult;
  result = allReduce(result, MPI_SUM, Communicator::communicator());
  return result;
}

template<typename Dimension>
int
DataBase<Dimension>::globalNumGhostNodes() const {
  int localResult = numGhostNodes();
  int result = localResult;
  result = allReduce(result, MPI_SUM, Communicator::communicator());
  return result;
}

template<typename Dimension>
int
DataBase<Dimension>::globalNumNodes() const {
  int localResult = numNodes();
  int result = localResult;
  result = allReduce(result, MPI_SUM, Communicator::communicator());
  return result;
}

//------------------------------------------------------------------------------
// Node iterators for all node IDs.
//------------------------------------------------------------------------------
template<typename Dimension>
AllNodeIterator<Dimension>
DataBase<Dimension>::nodeBegin() const {
  ConstNodeListIterator nodeListItr = nodeListBegin();
  while (nodeListItr < nodeListEnd() &&
	 (*nodeListItr)->numNodes() < 1) ++nodeListItr;
  CHECK(nodeListItr >= nodeListBegin() && nodeListItr <= nodeListEnd());
  return AllNodeIterator<Dimension>(nodeListItr,
                                    nodeListBegin(),
                                    nodeListEnd());
}

template<typename Dimension>
AllNodeIterator<Dimension>
DataBase<Dimension>::nodeEnd() const {
  return AllNodeIterator<Dimension>(nodeListEnd(),
                                    nodeListBegin(),
                                    nodeListEnd());
}

//------------------------------------------------------------------------------
// Node iterators for internal nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
InternalNodeIterator<Dimension>
DataBase<Dimension>::internalNodeBegin() const {
  ConstNodeListIterator nodeListItr = nodeListBegin();
  while (nodeListItr < nodeListEnd() &&
	 (*nodeListItr)->numInternalNodes() < 1) ++nodeListItr;
  CHECK(nodeListItr >= nodeListBegin() && nodeListItr <= nodeListEnd());
  return InternalNodeIterator<Dimension>(nodeListItr,
                                         nodeListBegin(),
                                         nodeListEnd());
}

template<typename Dimension>
InternalNodeIterator<Dimension>
DataBase<Dimension>::internalNodeEnd() const {
  return InternalNodeIterator<Dimension>(nodeListEnd(),
                                         nodeListBegin(),
                                         nodeListEnd());
}

//------------------------------------------------------------------------------
// Node iterators for ghost nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
GhostNodeIterator<Dimension>
DataBase<Dimension>::ghostNodeBegin() const {
  ConstNodeListIterator nodeListItr = nodeListBegin();
  while (nodeListItr < nodeListEnd() &&
	 (*nodeListItr)->numGhostNodes() < 1) ++nodeListItr;
  if (nodeListItr < nodeListEnd()) {
    return GhostNodeIterator<Dimension>(nodeListItr,
                                        nodeListBegin(),
                                        nodeListEnd(),
                                        (*nodeListItr)->firstGhostNode());
  } else {
    return this->ghostNodeEnd();
  }
}

template<typename Dimension>
GhostNodeIterator<Dimension>
DataBase<Dimension>::ghostNodeEnd() const {
  return GhostNodeIterator<Dimension>(nodeListEnd(),
                                      nodeListBegin(),
                                      nodeListEnd());
}

//------------------------------------------------------------------------------
// Node iterators for master neighbor nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
MasterNodeIterator<Dimension>
DataBase<Dimension>::masterNodeBegin(const std::vector<std::vector<int>>& masterLists) const {
  ConstNodeListIterator nodeListItr = nodeListBegin();
  unsigned iNodeList = 0;
  while (nodeListItr < nodeListEnd() &&
         masterLists[iNodeList].empty()) {
    ++nodeListItr;
    ++iNodeList;
  }
  if (nodeListItr < nodeListEnd()) {
    return MasterNodeIterator<Dimension>(nodeListItr,
                                         nodeListBegin(),
                                         nodeListEnd(),
                                         masterLists[iNodeList].begin(),
                                         masterLists);
  } else {
    return this->masterNodeEnd();
  }
}

template<typename Dimension>
MasterNodeIterator<Dimension>
DataBase<Dimension>::masterNodeEnd() const {
  return MasterNodeIterator<Dimension>(nodeListEnd(),
                                       nodeListBegin(),
                                       nodeListEnd(),
                                       std::vector<std::vector<int>>());
}

//------------------------------------------------------------------------------
// Node iterators for coarse neighbor nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
CoarseNodeIterator<Dimension>
DataBase<Dimension>::coarseNodeBegin(const std::vector<std::vector<int>>& coarseNeighbors) const {
  ConstNodeListIterator nodeListItr = nodeListBegin();
  unsigned iNodeList = 0;
  while (nodeListItr < nodeListEnd() &&
         coarseNeighbors[iNodeList].empty()) {
    ++nodeListItr;
    ++iNodeList;
  }
  if (nodeListItr < nodeListEnd()) {
    return CoarseNodeIterator<Dimension>(nodeListItr,
                                         nodeListBegin(),
                                         nodeListEnd(),
                                         coarseNeighbors[iNodeList].begin(),
                                         coarseNeighbors);
  } else {
    return this->coarseNodeEnd();
  }
}

template<typename Dimension>
CoarseNodeIterator<Dimension>
DataBase<Dimension>::coarseNodeEnd() const {
  return CoarseNodeIterator<Dimension>(nodeListEnd(),
                                       nodeListBegin(),
                                       nodeListEnd(),
                                       std::vector<std::vector<int>>());
}

//------------------------------------------------------------------------------
// Node iterators for refine neighbor nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
RefineNodeIterator<Dimension>
DataBase<Dimension>::refineNodeBegin(const std::vector<std::vector<int>>& refineNeighbors) const {
  ConstNodeListIterator nodeListItr = nodeListBegin();
  unsigned iNodeList = 0;
  while (nodeListItr < nodeListEnd() &&
         refineNeighbors[iNodeList].empty()) {
    ++nodeListItr;
    ++iNodeList;
  }
  if (nodeListItr < nodeListEnd()) {
    return RefineNodeIterator<Dimension>(nodeListItr,
                                         nodeListBegin(),
                                         nodeListEnd(),
                                         refineNeighbors[iNodeList].begin(),
                                         refineNeighbors);
  } else {
    return this->refineNodeEnd();
  }
}

template<typename Dimension>
RefineNodeIterator<Dimension>
DataBase<Dimension>::refineNodeEnd() const {
  return RefineNodeIterator<Dimension>(nodeListEnd(),
                                       nodeListBegin(),
                                       nodeListEnd(),
                                       std::vector<std::vector<int>>());
}

//------------------------------------------------------------------------------
// Node iterators for all fluid node IDs.
//------------------------------------------------------------------------------
template<typename Dimension>
AllNodeIterator<Dimension>
DataBase<Dimension>::fluidNodeBegin() const {
  ConstNodeListIterator nodeListItr = fluidNodeListAsNodeListBegin();
  while (nodeListItr < fluidNodeListAsNodeListEnd() &&
	 (*nodeListItr)->numNodes() < 1) ++nodeListItr;
  return AllNodeIterator<Dimension>(nodeListItr,
                                    fluidNodeListAsNodeListBegin(),
                                    fluidNodeListAsNodeListEnd());
}

template<typename Dimension>
AllNodeIterator<Dimension>
DataBase<Dimension>::fluidNodeEnd() const {
  return AllNodeIterator<Dimension>(fluidNodeListAsNodeListEnd(),
                                    fluidNodeListAsNodeListBegin(),
                                    fluidNodeListAsNodeListEnd());
}

//------------------------------------------------------------------------------
// Node iterators for fluid internal nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
InternalNodeIterator<Dimension>
DataBase<Dimension>::fluidInternalNodeBegin() const {
  ConstNodeListIterator nodeListItr = fluidNodeListAsNodeListBegin();
  while (nodeListItr < fluidNodeListAsNodeListEnd() &&
	 (*nodeListItr)->numInternalNodes() < 1) ++nodeListItr;
  return InternalNodeIterator<Dimension>(nodeListItr,
                                         fluidNodeListAsNodeListBegin(),
                                         fluidNodeListAsNodeListEnd());
}

template<typename Dimension>
InternalNodeIterator<Dimension>
DataBase<Dimension>::fluidInternalNodeEnd() const {
  return InternalNodeIterator<Dimension>(fluidNodeListAsNodeListEnd(),
                                         fluidNodeListAsNodeListBegin(),
                                         fluidNodeListAsNodeListEnd());
}

//------------------------------------------------------------------------------
// Node iterators for fluid ghost nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
GhostNodeIterator<Dimension>
DataBase<Dimension>::fluidGhostNodeBegin() const {
  ConstNodeListIterator nodeListItr = fluidNodeListAsNodeListBegin();
  while (nodeListItr < fluidNodeListAsNodeListEnd() &&
	 (*nodeListItr)->numGhostNodes() < 1) ++nodeListItr;
  CHECK(nodeListItr >= fluidNodeListAsNodeListBegin() && 
	nodeListItr <= fluidNodeListAsNodeListEnd());
  if (nodeListItr < fluidNodeListAsNodeListEnd()) {
    return GhostNodeIterator<Dimension>(nodeListItr,
                                        fluidNodeListAsNodeListBegin(),
                                        fluidNodeListAsNodeListEnd(),
                                        (*nodeListItr)->firstGhostNode());
  } else {
    return this->fluidGhostNodeEnd();
  }
}

template<typename Dimension>
GhostNodeIterator<Dimension>
DataBase<Dimension>::fluidGhostNodeEnd() const {
  return GhostNodeIterator<Dimension>(fluidNodeListAsNodeListEnd(),
                                      fluidNodeListAsNodeListBegin(),
                                      fluidNodeListAsNodeListEnd());
}

//------------------------------------------------------------------------------
// Node iterators for fluid master neighbor nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
MasterNodeIterator<Dimension>
DataBase<Dimension>::fluidMasterNodeBegin(const std::vector<std::vector<int>>& masterLists) const {
  ConstNodeListIterator nodeListItr = fluidNodeListAsNodeListBegin();
  unsigned iNodeList = 0;
  while (nodeListItr < fluidNodeListAsNodeListEnd() &&
         masterLists[iNodeList].empty()) {
    ++nodeListItr;
    ++iNodeList;
  }
  CHECK(nodeListItr >= fluidNodeListAsNodeListBegin() && 
	nodeListItr <= fluidNodeListAsNodeListEnd());
  if (nodeListItr < fluidNodeListAsNodeListEnd()) {
    return MasterNodeIterator<Dimension>(nodeListItr,
                                         fluidNodeListAsNodeListBegin(),
                                         fluidNodeListAsNodeListEnd(),
                                         masterLists[iNodeList].begin(),
                                         masterLists);
  } else {
    return this->fluidMasterNodeEnd();
  }
}

template<typename Dimension>
MasterNodeIterator<Dimension>
DataBase<Dimension>::fluidMasterNodeEnd() const {
  return MasterNodeIterator<Dimension>(fluidNodeListAsNodeListEnd(),
                                       fluidNodeListAsNodeListBegin(),
                                       fluidNodeListAsNodeListEnd(),
                                       std::vector<std::vector<int>>());
}

//------------------------------------------------------------------------------
// Node iterators for fluid coarse neighbor nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
CoarseNodeIterator<Dimension>
DataBase<Dimension>::fluidCoarseNodeBegin(const std::vector<std::vector<int>>& coarseNeighbors) const {
  ConstNodeListIterator nodeListItr = fluidNodeListAsNodeListBegin();
  unsigned iNodeList = 0;
  while (nodeListItr < fluidNodeListAsNodeListEnd() &&
         coarseNeighbors[iNodeList].empty()) {
    ++nodeListItr;
    ++iNodeList;
  }
  CHECK(nodeListItr >= fluidNodeListAsNodeListBegin() && 
	nodeListItr <= fluidNodeListAsNodeListEnd());
  if (nodeListItr < fluidNodeListAsNodeListEnd()) {
    return CoarseNodeIterator<Dimension>(nodeListItr,
                                         fluidNodeListAsNodeListBegin(),
                                         fluidNodeListAsNodeListEnd(),
                                         coarseNeighbors[iNodeList].begin(),
                                         coarseNeighbors);
  } else {
    return this->fluidCoarseNodeEnd();
  }
}

template<typename Dimension>
CoarseNodeIterator<Dimension>
DataBase<Dimension>::fluidCoarseNodeEnd() const {
  return CoarseNodeIterator<Dimension>(fluidNodeListAsNodeListEnd(),
                                       fluidNodeListAsNodeListBegin(),
                                       fluidNodeListAsNodeListEnd(),
                                       std::vector<std::vector<int>>());
}

//------------------------------------------------------------------------------
// Node iterators for fluid refine neighbor nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
RefineNodeIterator<Dimension>
DataBase<Dimension>::fluidRefineNodeBegin(const std::vector<std::vector<int>>& refineNeighbors) const {
  ConstNodeListIterator nodeListItr = fluidNodeListAsNodeListBegin();
  unsigned iNodeList = 0;
  while (nodeListItr < fluidNodeListAsNodeListEnd() &&
         refineNeighbors[iNodeList].empty()) {
    ++nodeListItr;
    ++iNodeList;
  }
  CHECK(nodeListItr >= fluidNodeListAsNodeListBegin() && 
	nodeListItr <= fluidNodeListAsNodeListEnd());
  if (nodeListItr < fluidNodeListAsNodeListEnd()) {
    return RefineNodeIterator<Dimension>(nodeListItr,
                                         fluidNodeListAsNodeListBegin(),
                                         fluidNodeListAsNodeListEnd(),
                                         refineNeighbors[iNodeList].begin(),
                                         refineNeighbors);
  } else {
    return this->fluidRefineNodeEnd();
  }
}

template<typename Dimension>
RefineNodeIterator<Dimension>
DataBase<Dimension>::fluidRefineNodeEnd() const {
  return RefineNodeIterator<Dimension>(fluidNodeListAsNodeListEnd(),
                                       fluidNodeListAsNodeListBegin(),
                                       fluidNodeListAsNodeListEnd(),
                                       std::vector<std::vector<int>>());
}

//------------------------------------------------------------------------------
// Optimize neighbor objects.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
reinitializeNeighbors() const {

  // Find the current bounding box and average node extent in one loop.
  // Compute the average node extent.
  auto xmin = Vector(std::numeric_limits<Scalar>::max()), xmax = Vector(std::numeric_limits<Scalar>::lowest());
  unsigned ntot = 0;
  Scalar havg = 0.0, hmax = 0.0, maxExtent = 0.0;
  for (auto itr = this->nodeListBegin(); itr != this->nodeListEnd(); ++itr) {
    const auto positions = (*itr)->positions();
    const auto  n = (*itr)->numInternalNodes();
    const auto& neighbor = (*itr)->neighbor();
    const auto  etaMax = neighbor.kernelExtent();
    maxExtent = std::max(maxExtent, etaMax);
    ntot += n;
    for (auto i = 0u; i < n; ++i) {
      const auto& xi = positions(i);
      xmin = elementWiseMin(xmin, xi);
      xmax = elementWiseMax(xmax, xi);
      const auto hi = neighbor.nodeExtent(i).maxElement()/etaMax;
      havg += hi;
      hmax = std::max(hmax, hi);
    }
  }

  // Find the global result across all processors.
  auto box = 0.0;
  for (auto i = 0; i != Dimension::nDim; ++i) {
    xmin(i) = allReduce(xmin(i), MPI_MIN, Communicator::communicator());
    xmax(i) = allReduce(xmax(i), MPI_MAX, Communicator::communicator());
    box = std::max(box, xmax(i) - xmin(i));
  }
  havg = allReduce(havg, MPI_SUM, Communicator::communicator());
  ntot = allReduce(ntot, MPI_SUM, Communicator::communicator());
  if (ntot > 0) havg /= ntot;
  hmax = allReduce(hmax, MPI_MAX, Communicator::communicator());

  box = std::max(box, maxExtent*hmax);
  for (auto i = 0; i < Dimension::nDim; ++i) {
    xmin(i) -= box;
    xmax(i) += box;
  }

  // Now initialize the neighbors.
  for (auto itr = this->nodeListBegin(); itr != this->nodeListEnd(); ++itr) {
    auto& neighbor = (*itr)->neighbor();
    neighbor.reinitialize(xmin, xmax, havg);
    neighbor.updateNodes();
  }
}

//------------------------------------------------------------------------------
// Update the connectivity map.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
updateConnectivityMap(const bool computeGhostConnectivity,
                      const bool computeOverlapConnectivity,
                      const bool computeIntersectionConnectivity) const {
  REQUIRE(mConnectivityMapPtr != 0 and
          mConnectivityMapPtr.get() != 0);
  mConnectivityMapPtr->rebuild(fluidNodeListBegin(), fluidNodeListEnd(),
                               computeGhostConnectivity, computeOverlapConnectivity, computeIntersectionConnectivity);
}

//------------------------------------------------------------------------------
// Patch the connectivity map.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
patchConnectivityMap(const FieldList<Dimension, int>& flags,
                     const FieldList<Dimension, int>& old2new) const {
  REQUIRE(mConnectivityMapPtr != 0 and
          mConnectivityMapPtr.get() != 0);
  mConnectivityMapPtr->patchConnectivity(flags, old2new);
}

//------------------------------------------------------------------------------
// Add a NodeList to this DataBase.
//------------------------------------------------------------------------------
// SolidNodeList
template<typename Dimension>
void
DataBase<Dimension>::
appendNodeList(SolidNodeList<Dimension>& nodeList) {
  REQUIRE(valid());
  if (haveNodeList(nodeList)) {
    cerr << "DataBase::appendNodeList() Warning: attempt to add SolidNodeList "
         << &nodeList << " to DataBase " << this
         << ", which already has it." << endl;
  } else {
    const NodeListRegistrar<Dimension>& nlr = NodeListRegistrar<Dimension>::instance();
    NodeListIterator orderItr = nlr.findInsertionPoint((NodeList<Dimension>*) &nodeList,
                                                       nodeListBegin(),
                                                       nodeListEnd());
    mNodeListPtrs.insert(orderItr, &nodeList);

    SolidNodeListIterator solidOrderItr = nlr.findInsertionPoint(&nodeList,
                                                                 solidNodeListBegin(),
                                                                 solidNodeListEnd());
    size_t delta = distance(solidNodeListBegin(), solidOrderItr);
    mSolidNodeListPtrs.insert(solidOrderItr, &nodeList);
    mSolidNodeListAsNodeListPtrs.insert(solidNodeListAsNodeListBegin() + delta,
                                        &nodeList);

    FluidNodeListIterator fluidOrderItr = nlr.findInsertionPoint(&nodeList,
                                                                 fluidNodeListBegin(),
                                                                 fluidNodeListEnd());
    delta = distance(fluidNodeListBegin(), fluidOrderItr);
    mFluidNodeListPtrs.insert(fluidOrderItr, &nodeList);
    mFluidNodeListAsNodeListPtrs.insert(fluidNodeListAsNodeListBegin() + delta,
                                        &nodeList);

  }
  ENSURE(valid());
}

// FluidNodeList
template<typename Dimension>
void
DataBase<Dimension>::
appendNodeList(FluidNodeList<Dimension>& nodeList) {
  REQUIRE(valid());
  if (haveNodeList(nodeList)) {
    cerr << "DataBase::appendNodeList() Warning: attempt to add FluidNodeList "
         << &nodeList << " to DataBase " << this
         << ", which already has it." << endl;
  } else {
    const NodeListRegistrar<Dimension>& nlr = NodeListRegistrar<Dimension>::instance();
    NodeListIterator orderItr = nlr.findInsertionPoint((NodeList<Dimension>*) &nodeList,
                                                       nodeListBegin(),
                                                       nodeListEnd());
    mNodeListPtrs.insert(orderItr, &nodeList);
    FluidNodeListIterator fluidOrderItr = nlr.findInsertionPoint(&nodeList,
                                                                 fluidNodeListBegin(),
                                                                 fluidNodeListEnd());
    const size_t delta = distance(fluidNodeListBegin(), fluidOrderItr);
    mFluidNodeListPtrs.insert(fluidOrderItr, &nodeList);
    mFluidNodeListAsNodeListPtrs.insert(fluidNodeListAsNodeListBegin() + delta,
                                        &nodeList);

  }
  ENSURE(valid());
}

// NodeList
template<typename Dimension>
void
DataBase<Dimension>::
appendNodeList(NodeList<Dimension>& nodeList) {
  REQUIRE(valid());
  if (haveNodeList(nodeList)) {
    cerr << "DataBase::appendNodeList() Warning: attempt to add NodeList "
         << &nodeList << " to DataBase " << this
         << ", which already has it." << endl;
  } else {
    const NodeListRegistrar<Dimension>& nlr = NodeListRegistrar<Dimension>::instance();
    NodeListIterator orderItr = nlr.findInsertionPoint(&nodeList,
                                                       nodeListBegin(),
                                                       nodeListEnd());
    mNodeListPtrs.insert(orderItr, &nodeList);
  }
  ENSURE(valid());
}

//------------------------------------------------------------------------------
// Delete a NodeList from this DataBase.
//------------------------------------------------------------------------------
// SolidNodeList
template<typename Dimension>
void
DataBase<Dimension>::
deleteNodeList(SolidNodeList<Dimension>& nodeList) {
  REQUIRE(valid());
  if (!haveNodeList(nodeList)) {
    cerr << "DataBase::deleteNodeList() Warning: attempt to remove SolidNodeList "
         << &nodeList << " from DataBase " << this
         << ", which does not have it." << endl;
  } else {
    // Erase from the NodeList vector.
    NodeListIterator nodeListItr = find(nodeListBegin(), nodeListEnd(),
                                        &nodeList);
    CHECK(nodeListItr != nodeListEnd());
    mNodeListPtrs.erase(nodeListItr);

    // Erase from the SolidNodeList vector.
    SolidNodeListIterator solidItr = find(solidNodeListBegin(), solidNodeListEnd(), &nodeList);
    CHECK(solidItr != solidNodeListEnd());
    mSolidNodeListPtrs.erase(solidItr);

    // Erase from the SolidNodeListAsNodeList vector.
    nodeListItr = find(solidNodeListAsNodeListBegin(),
                       solidNodeListAsNodeListEnd(),
                       &nodeList);
    CHECK(nodeListItr != solidNodeListAsNodeListEnd());
    mSolidNodeListAsNodeListPtrs.erase(nodeListItr);

    // Erase from the FluidNodeList vector.
    FluidNodeListIterator fluidItr = find(fluidNodeListBegin(), fluidNodeListEnd(), &nodeList);
    CHECK(fluidItr != fluidNodeListEnd());
    mFluidNodeListPtrs.erase(fluidItr);

    // Erase from the FluidNodeListAsNodeList vector.
    nodeListItr = find(fluidNodeListAsNodeListBegin(),
                       fluidNodeListAsNodeListEnd(),
                       &nodeList);
    CHECK(nodeListItr != fluidNodeListAsNodeListEnd());
    mFluidNodeListAsNodeListPtrs.erase(nodeListItr);
  }
  ENSURE(valid());
}

// FluidNodeList
template<typename Dimension>
void
DataBase<Dimension>::
deleteNodeList(FluidNodeList<Dimension>& nodeList) {
  REQUIRE(valid());
  if (!haveNodeList(nodeList)) {
    cerr << "DataBase::deleteNodeList() Warning: attempt to remove FluidNodeList "
         << &nodeList << " from DataBase " << this
         << ", which does not have it." << endl;
  } else {
    // Erase from the NodeList vector.
    NodeListIterator nodeListItr = find(nodeListBegin(), nodeListEnd(),
                                        &nodeList);
    CHECK(nodeListItr != nodeListEnd());
    mNodeListPtrs.erase(nodeListItr);

    // Erase from the FluidNodeList vector.
    FluidNodeListIterator fluidItr = find(fluidNodeListBegin(), fluidNodeListEnd(), &nodeList);
    CHECK(fluidItr != fluidNodeListEnd());
    mFluidNodeListPtrs.erase(fluidItr);

    // Erase from the FluidNodeListAsNodeList vector.
    nodeListItr = find(fluidNodeListAsNodeListBegin(),
                       fluidNodeListAsNodeListEnd(),
                       &nodeList);
    CHECK(nodeListItr != fluidNodeListAsNodeListEnd());
    mFluidNodeListAsNodeListPtrs.erase(nodeListItr);
  }
  ENSURE(valid());
}

// NodeList
template<typename Dimension>
void
DataBase<Dimension>::
deleteNodeList(NodeList<Dimension>& nodeList) {
  REQUIRE(valid());
  if (!haveNodeList(nodeList)) {
    cerr << "DataBase::deleteNodeList() Warning: attempt to remove NodeList "
         << &nodeList << " from DataBase " << this
         << ", which does not have it." << endl;
  } else {
    // Erase from the NodeList vector.
    NodeListIterator nodeListItr = find(nodeListBegin(), nodeListEnd(),
                                        &nodeList);
    CHECK(nodeListItr != nodeListEnd());
    mNodeListPtrs.erase(nodeListItr);
  }
  ENSURE(valid());
}

//------------------------------------------------------------------------------
// Test if the given NodeList is registered with this DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
DataBase<Dimension>::
haveNodeList(const NodeList<Dimension>& nodeList) const {
  ConstNodeListIterator itr = find(nodeListBegin(),
                                   nodeListEnd(),
                                   &nodeList);
  return itr != nodeListEnd();
}

//------------------------------------------------------------------------------
// Return the const list of NodeList pointers.
//------------------------------------------------------------------------------
template<typename Dimension>
const vector<NodeList<Dimension>*>&
DataBase<Dimension>::nodeListPtrs() const {
  return mNodeListPtrs;
}

//------------------------------------------------------------------------------
// Return the const list of FluidNodeList pointers.
//------------------------------------------------------------------------------
template<typename Dimension>
const vector<FluidNodeList<Dimension>*>&
DataBase<Dimension>::fluidNodeListPtrs() const {
  return mFluidNodeListPtrs;
}

//------------------------------------------------------------------------------
// Return the const list of SolidNodeList pointers.
//------------------------------------------------------------------------------
template<typename Dimension>
const vector<SolidNodeList<Dimension>*>&
DataBase<Dimension>::solidNodeListPtrs() const {
  return mSolidNodeListPtrs;
}

//------------------------------------------------------------------------------
// Set the master neighbor information for the NodeLists.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
setMasterNodeLists(const typename Dimension::Vector& position,
                   const typename Dimension::SymTensor& H,
                   std::vector<std::vector<int>>& masterLists,
                   std::vector<std::vector<int>>& coarseNeighbors,
                   const bool computeGhostConnectivity) const {
  Neighbor<Dimension>::setMasterNeighborGroup(position, H,
                                              nodeListBegin(),
                                              nodeListEnd(),
                                              maxKernelExtent(),
                                              masterLists,
                                              coarseNeighbors,
                                              computeGhostConnectivity);
}

//------------------------------------------------------------------------------
// Set the master neighbor information for the FluidNodeLists.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
setMasterFluidNodeLists(const typename Dimension::Vector& position,
                        const typename Dimension::SymTensor& H,
                        std::vector<std::vector<int>>& masterLists,
                        std::vector<std::vector<int>>& coarseNeighbors,
                        const bool computeGhostConnectivity) const {
  Neighbor<Dimension>::setMasterNeighborGroup(position, H,
                                              fluidNodeListBegin(),
                                              fluidNodeListEnd(),
                                              maxKernelExtent(),
                                              masterLists,
                                              coarseNeighbors,
                                              computeGhostConnectivity);
}

//------------------------------------------------------------------------------
// Set the refine neighbor information for the NodeLists.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
setRefineNodeLists(const typename Dimension::Vector& position,
                   const typename Dimension::SymTensor& H,
                   const std::vector<std::vector<int>>& coarseNeighbors,
                   std::vector<std::vector<int>>& refineNeighbors) const {
  REQUIRE(coarseNeighbors.size() == this->numNodeLists());
  refineNeighbors = std::vector<std::vector<int>>(this->numNodeLists());
  size_t iNodeList = 0;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr < nodeListEnd();
       ++nodeListItr, ++iNodeList) {
    (*nodeListItr)->neighbor().setRefineNeighborList(position, H, 
                                                     coarseNeighbors[iNodeList],
                                                     refineNeighbors[iNodeList]);
  }
}

//------------------------------------------------------------------------------
// Set the refine neighbor information for the FluidNodeLists.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
setRefineFluidNodeLists(const typename Dimension::Vector& position,
                        const typename Dimension::SymTensor& H,
                        const std::vector<std::vector<int>>& coarseNeighbors,
                        std::vector<std::vector<int>>& refineNeighbors) const {
  REQUIRE(coarseNeighbors.size() == this->numFluidNodeLists());
  refineNeighbors = std::vector<std::vector<int>>(this->numFluidNodeLists());
  size_t iNodeList = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd();
       ++nodeListItr, ++iNodeList) {
    (*nodeListItr)->neighbor().setRefineNeighborList(position, H,
                                                     coarseNeighbors[iNodeList],
                                                     refineNeighbors[iNodeList]);
  }
}

//------------------------------------------------------------------------------
// Return the global mass field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::globalMass() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr < nodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->mass());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the global position field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::globalPosition() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr < nodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->positions());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the global velocity field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::globalVelocity() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr < nodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->velocity());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the global H smoothing field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::SymTensor>
DataBase<Dimension>::globalHfield() const {
  REQUIRE(valid());
  FieldList<Dimension, SymTensor> result;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr < nodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->Hfield());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the global work field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::globalWork() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr < nodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->work());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the fluid mass field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::fluidMass() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->mass());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the fluid position field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::fluidPosition() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->positions());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the fluid velocity field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::fluidVelocity() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->velocity());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the fluid mass density field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::fluidMassDensity() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->massDensity());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the fluid specific thermal energy field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::fluidSpecificThermalEnergy() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->specificThermalEnergy());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the fluid H smoothing field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::SymTensor>
DataBase<Dimension>::fluidHfield() const {
  REQUIRE(valid());
  FieldList<Dimension, SymTensor> result;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->Hfield());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the fluid work field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::fluidWork() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->work());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid mass field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::solidMass() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->mass());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid position field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::solidPosition() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->positions());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid velocity field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::solidVelocity() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->velocity());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid mass density field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::solidMassDensity() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->massDensity());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid specific thermal energy field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::solidSpecificThermalEnergy() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->specificThermalEnergy());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid H smoothing field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::SymTensor>
DataBase<Dimension>::solidHfield() const {
  REQUIRE(valid());
  FieldList<Dimension, SymTensor> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->Hfield());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid work field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::solidWork() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->work());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid deviatoric stress field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::SymTensor>
DataBase<Dimension>::solidDeviatoricStress() const {
  REQUIRE(valid());
  FieldList<Dimension, SymTensor> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->deviatoricStress());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid plastic strain field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::solidPlasticStrain() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->plasticStrain());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid plastic strain rate field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Scalar>
DataBase<Dimension>::solidPlasticStrainRate() const {
  REQUIRE(valid());
  FieldList<Dimension, Scalar> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->plasticStrainRate());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid damage field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::SymTensor>
DataBase<Dimension>::solidDamage() const {
  REQUIRE(valid());
  FieldList<Dimension, SymTensor> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->damage());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid fragment ID field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, int>
DataBase<Dimension>::solidFragmentIDs() const {
  REQUIRE(valid());
  FieldList<Dimension, int> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->fragmentIDs());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the solid particle type field.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, int>
DataBase<Dimension>::solidParticleTypes() const {
  REQUIRE(valid());
  FieldList<Dimension, int> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->particleTypes());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the node extent for each NodeList.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::globalNodeExtent() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr < nodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->neighbor().nodeExtentField());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the node extent for each FluidNodeList.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::fluidNodeExtent() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr < fluidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->neighbor().nodeExtentField());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the node extent for each SolidNodeList.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, typename Dimension::Vector>
DataBase<Dimension>::solidNodeExtent() const {
  REQUIRE(valid());
  FieldList<Dimension, Vector> result;
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr < solidNodeListEnd(); ++nodeListItr) {
    result.appendField((*nodeListItr)->neighbor().nodeExtentField());
  }
  return result;
}

//------------------------------------------------------------------------------
// Return the global Hinverse field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
globalHinverse(FieldList<Dimension, typename Dimension::SymTensor>& result) const {
  REQUIRE(valid());
  this->resizeGlobalFieldList(result, SymTensor::zero);
  size_t nodeListi = 0;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr != nodeListEnd();
       ++nodeListItr, ++nodeListi) {
    (*nodeListItr)->Hinverse(*result[nodeListi]);
  }
}

//------------------------------------------------------------------------------
// Return the fluid Hinverse field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidHinverse(FieldList<Dimension, typename Dimension::SymTensor>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, SymTensor::zero);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++nodeListi) {
    (*nodeListItr)->Hinverse(*result[nodeListi]);
  }
}

//------------------------------------------------------------------------------
// Return the fluid pressure field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidPressure(FieldList<Dimension, typename Dimension::Scalar>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, 0.0, HydroFieldNames::pressure);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++nodeListi) {
    (*nodeListItr)->pressure(*result[nodeListi]);
  }
}

//------------------------------------------------------------------------------
// Return the fluid temperature field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidTemperature(FieldList<Dimension, typename Dimension::Scalar>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, 0.0, HydroFieldNames::temperature);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd(); 
       ++nodeListItr, ++nodeListi) {
    (*nodeListItr)->temperature(*result[nodeListi]);
  }
}

//------------------------------------------------------------------------------
// Return the fluid sound speed field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidSoundSpeed(FieldList<Dimension, typename Dimension::Scalar>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, 0.0, HydroFieldNames::soundSpeed);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++nodeListi) {
    (*nodeListItr)->soundSpeed(*result[nodeListi]);
  }
}

//------------------------------------------------------------------------------
// Return the fluid volume field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidVolume(FieldList<Dimension, typename Dimension::Scalar>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, 0.0, HydroFieldNames::volume, false);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++nodeListi) {
    (*nodeListItr)->volume(*result[nodeListi]);
  }
}

//------------------------------------------------------------------------------
// Return the fluid gamma (ratio of specific heats).
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidGamma(FieldList<Dimension, typename Dimension::Scalar>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, 0.0, HydroFieldNames::gamma, false);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++nodeListi) {
    const EquationOfState<Dimension>& eos = (*nodeListItr)->equationOfState();
    const Field<Dimension, Scalar>& rho = (*nodeListItr)->massDensity();
    const Field<Dimension, Scalar>& eps = (*nodeListItr)->specificThermalEnergy();
    Field<Dimension, Scalar>& gam = **result.fieldForNodeList(**nodeListItr);
    eos.setGammaField(gam, rho, eps);
  }
}

//------------------------------------------------------------------------------
// Return the fluid entropy.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidEntropy(FieldList<Dimension, typename Dimension::Scalar>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, 0.0, HydroFieldNames::entropy, false);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++nodeListi) {
    const EquationOfState<Dimension>& eos = (*nodeListItr)->equationOfState();
    const Field<Dimension, Scalar>& rho = (*nodeListItr)->massDensity();
    const Field<Dimension, Scalar>& eps = (*nodeListItr)->specificThermalEnergy();
    Field<Dimension, Scalar>& entropy = **result.fieldForNodeList(**nodeListItr);
    eos.setEntropy(entropy, rho, eps);
  }
}

//------------------------------------------------------------------------------
// Return the fluid linear momentum field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidLinearMomentum(FieldList<Dimension, typename Dimension::Vector>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, Vector::zero, HydroFieldNames::linearMomentum);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++nodeListi) {
    (*nodeListItr)->linearMomentum(*result[nodeListi]);
  }
}

//------------------------------------------------------------------------------
// Return the fluid energy field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidTotalEnergy(FieldList<Dimension, typename Dimension::Scalar>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, 0.0, HydroFieldNames::totalEnergy, false);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++nodeListi) {
    (*nodeListItr)->totalEnergy(*result[nodeListi]);
  }
}

//------------------------------------------------------------------------------
// Return the specific heat field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
fluidSpecificHeat(const FieldList<Dimension, typename Dimension::Scalar>& temperature,
                  FieldList<Dimension, typename Dimension::Scalar>& result) const {
  REQUIRE(valid());
  this->resizeFluidFieldList(result, 0.0, HydroFieldNames::specificHeat);
  size_t nodeListi = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd(); 
       ++nodeListItr, ++nodeListi) {
    const EquationOfState<Dimension>& eos = (*nodeListItr)->equationOfState();
    const Field<Dimension, Scalar>& rho = (*nodeListItr)->massDensity();
    const Field<Dimension, Scalar>& temp = **temperature.fieldForNodeList(**nodeListItr);
    Field<Dimension, Scalar>& specificHeat = **result.fieldForNodeList(**nodeListItr);
    eos.setSpecificHeat(specificHeat, rho, temp);
  }
}

//------------------------------------------------------------------------------
// Return the number of neighbors for each node based on the last update of 
// the connectivity map.
//------------------------------------------------------------------------------
template<typename Dimension>
FieldList<Dimension, int>
DataBase<Dimension>::numNeighbors() const {
  REQUIRE(valid());
  VERIFY(mConnectivityMapPtr != 0);
  FieldList<Dimension, int> result = newFluidFieldList(int(), "number of neighbors");
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd(); 
       ++nodeListItr) {
    const FluidNodeList<Dimension>& nodes = **nodeListItr;
    Field<Dimension, int>& count = **(result.fieldForNodeList(nodes));
    for (auto i = 0u; i != nodes.numInternalNodes(); ++i) {
      count(i) = 0;
      const vector< vector<int> >& connectivity = mConnectivityMapPtr->connectivityForNode(&nodes, i);
      for (typename vector< vector<int> >::const_iterator itr = connectivity.begin();
           itr != connectivity.end();
           ++itr) {
        count(i) += itr->size();
      }
    }
  }
  return result;
}

//------------------------------------------------------------------------------
// The maximum kernel extent being used.
//------------------------------------------------------------------------------
template<typename Dimension>
typename Dimension::Scalar
DataBase<Dimension>::
maxKernelExtent() const {
  Scalar result = 0.0;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr != nodeListEnd();
       ++nodeListItr) result = std::max(result, (**nodeListItr).neighbor().kernelExtent());
  return result;
}

//------------------------------------------------------------------------------
// Compute coordinates bounding all nodes in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
boundingBox(typename Dimension::Vector& xmin,
            typename Dimension::Vector& xmax,
            const bool ghost) const {

  // This method is now exported to an external stand-alone.  Maintained here
  // for backwards compatibility.
  const FieldList<Dimension, Vector> positions = this->globalPosition();
  Spheral::globalBoundingBox(positions, xmin, xmax, ghost);
}

//------------------------------------------------------------------------------
// Compute coordinates bounding all nodes in the DataBase.
// This version allows the user to mask off nodes (mask = 0 to ignore).
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
boundingBox(typename Dimension::Vector& xmin,
            typename Dimension::Vector& xmax,
            const FieldList<Dimension, int>& mask,
            const bool ghost) const {

  // Find our local bounds.
  xmin = DBL_MAX;
  xmax = -DBL_MAX;
  const FieldList<Dimension, Vector> positions = this->globalPosition();
  for (unsigned nodeList = 0; nodeList != positions.numFields(); ++nodeList) {
    const unsigned n = ghost ? positions[nodeList]->numElements() : positions[nodeList]->numInternalElements();
    for (unsigned i = 0; i != n; ++i) {
      if (mask(nodeList, i) != 0) {
        const Vector& xi = positions(nodeList, i);
        xmin = elementWiseMin(xmin, xi);
        xmax = elementWiseMax(xmax, xi);
      }
    }
  }

  // Now find the global bounds across all processors.
  for (int i = 0; i != Dimension::nDim; ++i) {
    xmin(i) = allReduce(xmin(i), MPI_MIN, Communicator::communicator());
    xmax(i) = allReduce(xmax(i), MPI_MAX, Communicator::communicator());
  }
}

//------------------------------------------------------------------------------
// Return the local (per domain) sampling extents.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
localSamplingBoundingVolume(typename Dimension::Vector& centroid,
			    double& radiusNodes,
                            double& radiusSample,
			    typename Dimension::Vector& xminNodes,
                            typename Dimension::Vector& xmaxNodes,
			    typename Dimension::Vector& xminSample,
                            typename Dimension::Vector& xmaxSample) const {

  // Find our local bounds.
  centroid = Vector();
  xminNodes = FLT_MAX;
  xmaxNodes = -FLT_MAX;
  xminSample = FLT_MAX;
  xmaxSample = -FLT_MAX;
  size_t count = 0;
  const FieldList<Dimension, Vector> positions = this->globalPosition();
  const FieldList<Dimension, Vector> extent = this->globalNodeExtent();
  for (auto nodeList = 0u; nodeList != positions.numFields(); ++nodeList) {
    for (auto i = 0u; i != mNodeListPtrs[nodeList]->numInternalNodes(); ++i) {
      const Vector& xi = positions(nodeList, i);
      const Vector& extenti = extent(nodeList, i);
      const Vector xmini = xi - extenti;
      const Vector xmaxi = xi + extenti;
      centroid += xi;
      xminNodes = elementWiseMin(xminNodes, xi);
      xmaxNodes = elementWiseMax(xmaxNodes, xi);
      xminSample = elementWiseMin(xminSample, xmini);
      xmaxSample = elementWiseMax(xmaxSample, xmaxi);
      ++count;
    }
  }

  // Normalize the centroid.
  if (count > 0) centroid /= count;

  // Find the maximal radial extent from the centroid.
  radiusNodes = 0.0;
  radiusSample = 0.0;
  FieldList<Dimension, SymTensor> Hinv = newGlobalFieldList(SymTensor::zero);
  this->globalHinverse(Hinv);
  for (auto nodeList = 0u; nodeList != positions.numFields(); ++nodeList) {
    for (auto i = 0u; i != mNodeListPtrs[nodeList]->numInternalNodes(); ++i) {
      const Vector& xi = positions(nodeList, i);
      const Vector dr = xi - centroid;
      const double drMag = dr.magnitude();
      const double hi = (Hinv(nodeList, i)*dr).magnitude() * safeInv(drMag, 1.0e-20);
      radiusNodes = max(radiusNodes, drMag);
      radiusSample = max(radiusSample, drMag + 2.0*hi);
    }
  }

  // Puff things out a tiny bit in an effort to make sure we're domain 
  // decomposition independent.
  const Vector delta = 0.001*(xmaxSample - xminSample);
  xminNodes -= delta;
  xmaxNodes += delta;
  xminSample -= delta;
  xmaxSample += delta;
  radiusNodes *= 1.001;
  radiusSample *= 1.001;
}

//------------------------------------------------------------------------------
// Return the global min and max sampling extents.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
globalSamplingBoundingVolume(typename Dimension::Vector& centroid,
			     double& radiusNodes,
			     double& radiusSample,
			     typename Dimension::Vector& xminNodes,
                             typename Dimension::Vector& xmaxNodes,
			     typename Dimension::Vector& xminSample,
                             typename Dimension::Vector& xmaxSample) const {

  // Find our local bounds.
  this->localSamplingBoundingVolume(centroid, radiusNodes, radiusSample, 
				    xminNodes, xmaxNodes,
				    xminSample, xmaxSample);

#ifdef USE_MPI
  // Now find the global bounds across all processors.
  {
    size_t nlocal = this->numInternalNodes();
    centroid *= nlocal;
    for (int i = 0; i != Dimension::nDim; ++i) {
      xminNodes(i) = allReduce(xminNodes(i), MPI_MIN, Communicator::communicator());
      xmaxNodes(i) = allReduce(xmaxNodes(i), MPI_MAX, Communicator::communicator());
      xminSample(i) = allReduce(xminSample(i), MPI_MIN, Communicator::communicator());
      xmaxSample(i) = allReduce(xmaxSample(i), MPI_MAX, Communicator::communicator());
      centroid(i) = allReduce(centroid(i), MPI_SUM, Communicator::communicator());
    }

    // Fix up the centroid and radii.
    size_t nglobal = allReduce((uint64_t) nlocal, MPI_SUM, Communicator::communicator());
    if (nglobal > 0) {
      centroid /= nglobal;
      radiusNodes = 0.0;
      radiusSample = 0.0;
      const FieldList<Dimension, Vector> positions = this->globalPosition();
      const FieldList<Dimension, Vector> extent = this->globalNodeExtent();
      FieldList<Dimension, SymTensor> Hinv = this->newGlobalFieldList(SymTensor::zero, "H inverse");
      this->globalHinverse(Hinv);
      for (auto nodeList = 0u; nodeList != positions.numFields(); ++nodeList) {
	for (auto i = 0u; i != mNodeListPtrs[nodeList]->numInternalNodes(); ++i) {
	  const Vector& xi = positions(nodeList, i);
	  const Vector dr = xi - centroid;
	  const Vector drUnit = dr.unitVector();
	  const double drMag = dr.magnitude();
	  const double hi = (Hinv(nodeList, i)*drUnit).magnitude();
	  radiusNodes = max(radiusNodes, drMag);
	  radiusSample = max(radiusSample, drMag + 2.0*hi);
	}
      }
      radiusNodes = allReduce(radiusNodes, MPI_MAX, Communicator::communicator());
      radiusSample = allReduce(radiusSample, MPI_MAX, Communicator::communicator());
      const Vector delta = 0.001*(xmaxSample - xminSample);
      radiusNodes *= 1.001;
      radiusSample *= 1.001;
    }
  }
#endif
}

//------------------------------------------------------------------------------
// Return the local (per domain) min and max sampling extents for nodes that
// are connected.  This could represent an arbitrary number of results (in the
// extreme case where all nodes a disconnected you get back a set the size of 
// the number of nodes!)
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
localSamplingBoundingBoxes(vector<typename Dimension::Vector>& xminima,
                           vector<typename Dimension::Vector>& xmaxima) const {

  // Initialize the result.
  xminima = vector<Vector>();
  xmaxima = vector<Vector>();

  // We use our connectivity to make this more efficient.
  this->updateConnectivityMap(false, false, false);
  const ConnectivityMap<Dimension>& connectivityMap = this->connectivityMap(false, false, false);
  const FieldList<Dimension, Vector> positions = this->globalPosition();
  const FieldList<Dimension, Vector> extent = this->globalNodeExtent();
  const int numNodeLists = this->numNodeLists();

  // Iterate over the nodes and progressively build up the result by linking
  // things together as they're connected.
  FieldList<Dimension, int> flags = this->newGlobalFieldList(0, "flags");
  for (int nodeListi = 0; nodeListi != numNodeLists; ++nodeListi) {
    const NodeList<Dimension>& nodeList = positions[nodeListi]->nodeList();
    for (auto i = 0u; i != nodeList.numInternalNodes(); ++i) {
      if (flags(nodeListi, i) == 0) {
        const Vector& xi = positions(nodeListi, i);
        const Vector& extenti = extent(nodeListi, i);
        Vector xminGroup = xi - extenti;
        Vector xmaxGroup = xi + extenti;
        flags(nodeListi, i) = 1;

        // Find the min/max extent of this connected set of nodes.
        const vector<vector<int> >& fullConnectivity = connectivityMap.connectivityForNode(&nodeList, i);
        CHECK((int)fullConnectivity.size() == numNodeLists);
        for (int nodeListj = 0; nodeListj != numNodeLists; ++nodeListj) {
          const vector<int>& connectivity = fullConnectivity[nodeListj];
          for (vector<int>::const_iterator jItr = connectivity.begin();
               jItr != connectivity.end();
               ++jItr) {
            const int j = *jItr;
            const Vector& xj = positions(nodeListj, j);
            const Vector& extentj = extent(nodeListj, j);
            xminGroup = elementWiseMin(xminGroup, xj - extentj);
            xmaxGroup = elementWiseMax(xmaxGroup, xj + extentj);
            flags(nodeListj, j) = 1;
          }
        }

        // Compare this batches min/max extents with the results so far.
        CHECK(xminima.size() == xmaxima.size());
        unsigned int k = 0;
        bool matched = false;
        while (k != xminima.size() and not matched) {
          if (testBoxIntersection(xminGroup, xmaxGroup, xminima[k], xmaxima[k])) {
            xminima[k] = elementWiseMin(xminima[k], xminGroup);
            xmaxima[k] = elementWiseMax(xmaxima[k], xmaxGroup);
            matched = true;
          }
          ++k;
        }
        if (not matched) {
          xminima.push_back(xminGroup);
          xmaxima.push_back(xmaxGroup);
        }
      }
    }
  }

  // Look for any overlaps between the boxes.
  int ibox = xminima.size() - 1;
  while (ibox > 0) {
    int jbox = 0;
    while (jbox != ibox) {
      if (testBoxIntersection(xminima[ibox], xmaxima[ibox], xminima[jbox], xmaxima[jbox])) {
        xminima[jbox] = elementWiseMin(xminima[jbox], xminima[ibox]);
        xmaxima[jbox] = elementWiseMax(xmaxima[jbox], xmaxima[ibox]);
        xminima.erase(xminima.begin() + ibox);
        xmaxima.erase(xmaxima.begin() + ibox);
        jbox = ibox;
      } else {
        ++jbox;
      }
    }
    --ibox;
  }

}

//------------------------------------------------------------------------------
// The global version of above.
//------------------------------------------------------------------------------
template<typename Dimension>
void
DataBase<Dimension>::
globalSamplingBoundingBoxes(vector<typename Dimension::Vector>& xminima,
                            vector<typename Dimension::Vector>& xmaxima) const {

  // First get each domains local values.
  localSamplingBoundingBoxes(xminima, xmaxima);

#ifdef USE_MPI
  // Parallel crap.
  const int procID = Process::getRank();
  const int numProcs = Process::getTotalNumberOfProcesses();

  // Get each domains min & maxes.
  vector<char> localBuffer;
  packElement(xminima, localBuffer);
  packElement(xmaxima, localBuffer);
  xminima = vector<Vector>();
  xmaxima = vector<Vector>();
  for (int sendProc = 0; sendProc != numProcs; ++sendProc) {
    vector<char> buffer = localBuffer;
    int bufSize = localBuffer.size();
    MPI_Bcast(&bufSize, 1, MPI_INT, sendProc, Communicator::communicator());
    if (procID != sendProc) buffer.resize(bufSize);
    MPI_Bcast(&buffer.front(), bufSize, MPI_CHAR, sendProc, Communicator::communicator());
    vector<Vector> xminimaDomain, xmaximaDomain;
    vector<char>::const_iterator bufItr = buffer.begin();
    unpackElement(xminimaDomain, bufItr, buffer.end());
    unpackElement(xmaximaDomain, bufItr, buffer.end());
    CHECK(bufItr == buffer.end());
    copy(xminimaDomain.begin(), xminimaDomain.end(), back_inserter(xminima));
    copy(xmaximaDomain.begin(), xmaximaDomain.end(), back_inserter(xmaxima));
  }

  // Go through the boxes for each domain, joining them up where possible.
  int ibox = xminima.size() - 1;
  while (ibox > 0) {
    int jbox = 0;
    while (jbox != ibox) {
      if (testBoxIntersection(xminima[ibox], xmaxima[ibox], xminima[jbox], xmaxima[jbox])) {
        xminima[jbox] = elementWiseMin(xminima[jbox], xminima[ibox]);
        xmaxima[jbox] = elementWiseMax(xmaxima[jbox], xmaxima[ibox]);
        xminima.erase(xminima.begin() + ibox);
        xmaxima.erase(xmaxima.begin() + ibox);
        jbox = ibox;
      } else {
        ++jbox;
      }
    }
    --ibox;
  }
#endif
}

//------------------------------------------------------------------------------
// Test if the DataBase is "valid", or internally consistent.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
DataBase<Dimension>::valid() const {

  bool ok = (numNodeLists() >= numFluidNodeLists());

  // Verify that all FluidNodeLists are listed in the NodeList array.
  ConstFluidNodeListIterator fluidNodeItr = fluidNodeListBegin();
  while (ok && fluidNodeItr < fluidNodeListEnd()) {
    ok = haveNodeList(dynamic_cast<const NodeList<Dimension>&>(**fluidNodeItr));
    ++fluidNodeItr;
  }

  // Verify that all FluidNodeLists are listed in the FluidNodeListAsNodeList
  // array.
  fluidNodeItr = fluidNodeListBegin();
  while (ok && fluidNodeItr < fluidNodeListEnd()) {
    ok = find(mFluidNodeListAsNodeListPtrs.begin(),
              mFluidNodeListAsNodeListPtrs.end(),
              *fluidNodeItr) != mFluidNodeListAsNodeListPtrs.end();
    ++fluidNodeItr;
  }

  // Verify that all SolidNodeLists are listed in the NodeList array.
  ConstSolidNodeListIterator solidNodeItr = solidNodeListBegin();
  while (ok && solidNodeItr < solidNodeListEnd()) {
    ok = haveNodeList(dynamic_cast<const NodeList<Dimension>&>(**solidNodeItr));
    ++solidNodeItr;
  }

  // Verify that all SolidNodeLists are listed in the SolidNodeListAsNodeList
  // array.
  solidNodeItr = solidNodeListBegin();
  while (ok && solidNodeItr < solidNodeListEnd()) {
    ok = find(mSolidNodeListAsNodeListPtrs.begin(),
              mSolidNodeListAsNodeListPtrs.end(),
              *solidNodeItr) != mSolidNodeListAsNodeListPtrs.end();
    ++solidNodeItr;
  }

  // Verify that the NodeLists are listed in the proper order.
  const NodeListRegistrar<Dimension>& nlr = NodeListRegistrar<Dimension>::instance();
  if (ok) {
    typename NodeListRegistrar<Dimension>::const_iterator checkItr = nlr.begin();
    ConstNodeListIterator itr = nodeListBegin();
    while (itr != nodeListEnd()) {
      while (checkItr != nlr.end() && *checkItr != *itr)
        ++checkItr;
      ++itr;
    }
    ok = (itr == nodeListEnd());
  }

  // Verify that the FluidNodeLists are listed in the proper order.
  if (ok) {
    typename NodeListRegistrar<Dimension>::const_iterator checkItr = nlr.begin();
    ConstFluidNodeListIterator itr = fluidNodeListBegin();
    while (itr != fluidNodeListEnd()) {
      while (checkItr != nlr.end() && *checkItr != dynamic_cast<NodeList<Dimension>*>(*itr))
        ++checkItr;
      ++itr;
    }
    ok = (itr == fluidNodeListEnd());
  }

  // Verify that the FluidNodeListsAsNodeLists are listed in the proper order.
  if (ok) {
    typename NodeListRegistrar<Dimension>::const_iterator checkItr = nlr.begin();
    ConstNodeListIterator itr = fluidNodeListAsNodeListBegin();
    while (itr != fluidNodeListAsNodeListEnd()) {
      while (checkItr != nlr.end() && *checkItr != *itr)
        ++checkItr;
      ++itr;
    }
    ok = (itr == fluidNodeListAsNodeListEnd());
  }

  // Verify that the SolidNodeLists are listed in the proper order.
  if (ok) {
    typename NodeListRegistrar<Dimension>::const_iterator checkItr = nlr.begin();
    ConstSolidNodeListIterator itr = solidNodeListBegin();
    while (itr != solidNodeListEnd()) {
      while (checkItr != nlr.end() && *checkItr != dynamic_cast<NodeList<Dimension>*>(*itr))
        ++checkItr;
      ++itr;
    }
    ok = (itr == solidNodeListEnd());
  }

  // Verify that the SolidNodeListsAsNodeLists are listed in the proper order.
  if (ok) {
    typename NodeListRegistrar<Dimension>::const_iterator checkItr = nlr.begin();
    ConstNodeListIterator itr = solidNodeListAsNodeListBegin();
    while (itr != solidNodeListAsNodeListEnd()) {
      while (checkItr != nlr.end() && *checkItr != *itr)
        ++checkItr;
      ++itr;
    }
    ok = (itr == solidNodeListAsNodeListEnd());
  }

  return ok;
}

//------------------------------------------------------------------------------
// Set the static dimensionality.
//------------------------------------------------------------------------------
template<typename Dimension>
int DataBase<Dimension>::nDim = Dimension::nDim;

template<typename Dimension>
bool DataBase<Dimension>::isRZ = false;

}

