//---------------------------------Spheral++----------------------------------//
// Integrator -- The topmost abstract base class for all integrator classes
// Spheral++.  Integrator classes take a list of Physics packages and advance
// them in time.
//
// Created by JMO, Wed May 31 21:58:08 PDT 2000
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "DataOutput/Restart.hh"
#include "DataBase/DataBase.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "Field/FieldList.hh"
#include "Physics/Physics.hh"
#include "Boundary/Boundary.hh"
#include "Hydro/HydroFieldNames.hh"
// #include "Utilities/timingUtilities.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Utilities/allReduce.hh"
#include "Distributed/Communicator.hh"
#include "Utilities/DBC.hh"
#include "Integrator.hh"

#include <limits.h>
#include <float.h>
#include <algorithm>
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
// Empty constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
Integrator<Dimension>::Integrator():
  mDtMin(0.0),
  mDtMax(FLT_MAX),
  mDtGrowth(2.0),
  mLastDt(1e-5),
  mDtMultiplier(1.0),
  mDtCheckFrac(0.5),
  mCurrentTime(0.0),
  mCurrentCycle(0),
  mUpdateBoundaryFrequency(1),
  mVerbose(false),
  mAllowDtCheck(false),
  mRequireConnectivity(true),
  mRequireGhostConnectivity(false),
  mRequireOverlapConnectivity(false),
  mRequireIntersectionConnectivity(false),
  mDataBasePtr(0),
  mPhysicsPackages(0),
  mRigorousBoundaries(false),
  mCullGhostNodes(true),
  mRestart(registerWithRestart(*this)) {
}

//------------------------------------------------------------------------------
// Construct with the given DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
Integrator<Dimension>::
Integrator(DataBase<Dimension>& dataBase):
  mDtMin(0.0),
  mDtMax(FLT_MAX),
  mDtGrowth(2.0),
  mLastDt(1e-5),
  mDtMultiplier(1.0),
  mDtCheckFrac(0.5),
  mCurrentTime(0.0),
  mCurrentCycle(0),
  mUpdateBoundaryFrequency(1),
  mVerbose(false),
  mAllowDtCheck(false),
  mRequireConnectivity(true),
  mRequireGhostConnectivity(false),
  mRequireOverlapConnectivity(false),
  mRequireIntersectionConnectivity(false),
  mDataBasePtr(&dataBase),
  mPhysicsPackages(0),
  mRigorousBoundaries(false),
  mCullGhostNodes(true),
  mRestart(registerWithRestart(*this)) {
}

//------------------------------------------------------------------------------
// Construct with the given DataBase and Physics packages.
//------------------------------------------------------------------------------
template<typename Dimension>
Integrator<Dimension>::
Integrator(DataBase<Dimension>& dataBase,
           const vector<Physics<Dimension>*>& physicsPackages):
  mDtMin(0.0),
  mDtMax(FLT_MAX),
  mDtGrowth(2.0),
  mLastDt(1e-5),
  mDtMultiplier(1.0),
  mDtCheckFrac(0.5),
  mCurrentTime(0.0),
  mCurrentCycle(0),
  mUpdateBoundaryFrequency(1),
  mVerbose(false),
  mAllowDtCheck(false),
  mRequireConnectivity(true),
  mRequireGhostConnectivity(false),
  mDataBasePtr(&dataBase),
  mPhysicsPackages(physicsPackages),
  mRigorousBoundaries(false),
  mCullGhostNodes(true),
  mRestart(registerWithRestart(*this)) {
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
Integrator<Dimension>::~Integrator() {
}

//------------------------------------------------------------------------------
// Assignment
//------------------------------------------------------------------------------
template<typename Dimension>
Integrator<Dimension>&
Integrator<Dimension>::
operator=(const Integrator<Dimension>& rhs) {
  if (this != &rhs) {
    mDtMin = rhs.mDtMin;
    mDtMax = rhs.mDtMax;
    mDtGrowth = rhs.mDtGrowth;
    mLastDt = rhs.mLastDt;
    mDtMultiplier = rhs.mDtMultiplier;
    mDtCheckFrac = rhs.mDtCheckFrac;
    mCurrentTime = rhs.mCurrentTime;
    mCurrentCycle = rhs.mCurrentCycle;
    mDataBasePtr = rhs.mDataBasePtr;
    mPhysicsPackages = rhs.mPhysicsPackages;
    mRigorousBoundaries = rhs.mRigorousBoundaries;
    mUpdateBoundaryFrequency = rhs.mUpdateBoundaryFrequency;
    mCullGhostNodes = rhs.mCullGhostNodes;
    mVerbose = rhs.mVerbose;
    mAllowDtCheck = rhs.mAllowDtCheck;
    mRequireConnectivity = rhs.mRequireConnectivity;
    mRequireGhostConnectivity = rhs.mRequireGhostConnectivity;
    mRequireOverlapConnectivity = rhs.mRequireOverlapConnectivity;
    mRequireIntersectionConnectivity = rhs.mRequireIntersectionConnectivity;
  }
  return *this;
}

//------------------------------------------------------------------------------
// step
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Integrator<Dimension>::
step(const typename Dimension::Scalar maxTime) {
  DataBase<Dimension>& db = this->accessDataBase();
  State<Dimension> state(db, this->physicsPackagesBegin(), this->physicsPackagesEnd());
  StateDerivatives<Dimension> derivs(db, this->physicsPackagesBegin(), this->physicsPackagesEnd());
  auto success = false;
  auto count = 0;
  auto maxIterations = 10;
  while (not success and count < maxIterations) {
    ++count;
    if (count == maxIterations) mAllowDtCheck = false;
    success = this->step(maxTime, state, derivs);
    if (count == maxIterations) mAllowDtCheck = true;
    if (not success) {
      if (Process::getRank() == 0) {
        cerr << "Integrator::step reported unstable timestep -- cutting dt and trying again: " << count << "/10" << endl;
        mDtMultiplier *= 0.5;
      }
    }
  }
  mDtMultiplier = 1.0;
  return success;
}

//------------------------------------------------------------------------------
// Loop over the stored physics packages and pick the minimum timestep.
//------------------------------------------------------------------------------
template<typename Dimension>
typename Dimension::Scalar
Integrator<Dimension>::
selectDt(const typename Dimension::Scalar dtMin,
         const typename Dimension::Scalar dtMax,
         const State<Dimension>& state,
         const StateDerivatives<Dimension>& derivs) const {

  REQUIRE(dtMin >= 0 and dtMax > 0);
  REQUIRE(dtMin <= dtMax);

  typedef typename Physics<Dimension>::TimeStepType TimeStepType;

  // Get the current time and data base.
  auto        t = currentTime();
  const auto& db = dataBase();

  // Loop over each package, and pick their timesteps.
  TimeStepType dt(dtMax, "");
  for (auto physicsItr = physicsPackagesBegin(); physicsItr < physicsPackagesEnd(); ++physicsItr) {
    auto dtVote = (*physicsItr)->dt(db, state, derivs, t);
    if (dtVote.first > 0.0 and dtVote.first < dt.first) dt = dtVote;
  }

  // Apply any dt scaling due to iteration
  dt.first *= mDtMultiplier;

  // We also require that the timestep is not allowed to grow faster than a
  // prescribed fraction.
  dt.first = std::min(dt.first, dtGrowth()*lastDt());

  // Enforce the timestep boundaries.
  dt.first = std::min(dtMax, std::max(dtMin, dt.first));

  CHECK(dt.first >= 0.0 and
        dt.first >= dtMin and dt.first <= dtMax);

  // In the parallel case we need to find the minimum timestep across all processors.
  const auto globalDt = allReduce(dt.first, MPI_MIN, Communicator::communicator());

  // Are we verbose?
  if (dt.first == globalDt and 
      (verbose() or globalDt < mDtMin)) {
    cout << "Selected timestep of "
         << dt.first << endl
         << dt.second << endl;
  }
  cout.flush();
  dt.first = globalDt;

  return dt.first;
}

//------------------------------------------------------------------------------
// Initializations for integrators.  To be called once at the beginning of an
// integrator step.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::preStepInitialize(State<Dimension>& state,
                                         StateDerivatives<Dimension>& derivs) {

  // Check if we need to construct connectivity.
  mRequireConnectivity = false;
  mRequireGhostConnectivity = false;
  mRequireOverlapConnectivity = false;
  mRequireIntersectionConnectivity = false;
  for (typename Integrator<Dimension>::ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {
    mRequireConnectivity = (mRequireConnectivity or (*physicsItr)->requireConnectivity());
    mRequireGhostConnectivity = (mRequireGhostConnectivity or (*physicsItr)->requireGhostConnectivity());
    mRequireOverlapConnectivity = (mRequireOverlapConnectivity or (*physicsItr)->requireOverlapConnectivity());
    mRequireIntersectionConnectivity = (mRequireIntersectionConnectivity or (*physicsItr)->requireIntersectionConnectivity());
  }

  // Intialize neighbors if need be.
  DataBase<Dimension>& db = accessDataBase();
  // if (mRequireConnectivity) db.reinitializeNeighbors();

  // Set the boundary conditions.
  if ((not mRigorousBoundaries) and (mCurrentCycle % mUpdateBoundaryFrequency == 0)) {
    setGhostNodes();
  }
  applyGhostBoundaries(state, derivs);

  // Register the now updated connectivity with the state.
  if (mRequireConnectivity) {
    state.enrollConnectivityMap(db.connectivityMapPtr(mRequireGhostConnectivity, mRequireOverlapConnectivity, mRequireIntersectionConnectivity));
  }

  // Loop over the physics packages and perform any necessary initializations.
  for (typename Integrator<Dimension>::ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {
    (*physicsItr)->preStepInitialize(db, state, derivs);
  }
}

//------------------------------------------------------------------------------
// Initialize all physics packages before evaluating derivatives.
// Called before each call to Physics::evaluateDerivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::initializeDerivatives(const double t,
                                             const double dt,
                                             State<Dimension>& state,
                                             StateDerivatives<Dimension>& derivs) {

  // Initialize the work fields.
  DataBase<Dimension>& db = accessDataBase();
  for (typename DataBase<Dimension>::NodeListIterator nodeListItr = db.nodeListBegin();
       nodeListItr < db.nodeListEnd();
       ++nodeListItr) {
    (*nodeListItr)->work() = 0.0;
  }

  // Loop over the physics packages and perform any necessary initializations.
  for (typename Integrator<Dimension>::ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {
    (*physicsItr)->initialize(t, dt, db, state, derivs);
  }

  // Physics packages may have called boundary conditions as well, so finalize any
  // outstanding boundary conditions here.
  this->finalizeGhostBoundaries();
}

//------------------------------------------------------------------------------
// Iterate over all physics packages and call evaluateDerivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::evaluateDerivatives(const Scalar t,
                                           const Scalar dt,
                                           const DataBase<Dimension>& dataBase,
                                           const State<Dimension>& state,
                                           StateDerivatives<Dimension>& derivs) const {

  // Loop over the physics packages and have them evaluate their derivatives.
  for (typename Integrator<Dimension>::ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {
    (*physicsItr)->evaluateDerivatives(t, dt, dataBase, state, derivs);
  }
}

//------------------------------------------------------------------------------
// Iterate over all physics packages and call finalizeDerivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::finalizeDerivatives(const Scalar t,
                                           const Scalar dt,
                                           const DataBase<Dimension>& dataBase,
                                           const State<Dimension>& state,
                                           StateDerivatives<Dimension>& derivs) const {

  // Loop over the physics packages and have them finalize their derivatives.
  for (typename Integrator<Dimension>::ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {
    (*physicsItr)->finalizeDerivatives(t, dt, dataBase, state, derivs);
  }
}

//------------------------------------------------------------------------------
// Iterate over all physics packages and let them do any post state update 
// stuff.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::postStateUpdate(const Scalar t,
                                       const Scalar dt,
                                       const DataBase<Dimension>& dataBase, 
                                       State<Dimension>& state,
                                       StateDerivatives<Dimension>& derivs) const {

  // Loop over the physics packages.
  for (typename Integrator<Dimension>::ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {
    (*physicsItr)->postStateUpdate(t, dt, dataBase, state, derivs);
  }
}

//------------------------------------------------------------------------------
// Finalize at the completion of a time step.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::postStepFinalize(const double t,
                                        const double dt,
                                        State<Dimension>& state,
                                        StateDerivatives<Dimension>& derivs) {

  // Loop over the physics packages and perform any necessary finalizations.
  DataBase<Dimension>& db = accessDataBase();
//   for (typename DataBase<Dimension>::FluidNodeListIterator nodeListItr = db.fluidNodeListBegin();
//        nodeListItr != db.fluidNodeListEnd(); 
//        ++nodeListItr) {
//     (*nodeListItr)->neighbor().updateNodes();
//   }
  for (typename Integrator<Dimension>::ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {
    (*physicsItr)->finalize(t, dt, db, state, derivs);
  }
}

//------------------------------------------------------------------------------
// Add a physics package.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::
appendPhysicsPackage(Physics<Dimension>& package) {
  if (!havePhysicsPackage(package)) {
    mPhysicsPackages.push_back(&package);
  } else {
    cerr << "Warning: attempt to append Physics package " << &package
         << "to Integrator " << this << " which already has it." << endl;
  }
}

//------------------------------------------------------------------------------
// Reset the Physics packages to a new set
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::
resetPhysicsPackages(std::vector<Physics<Dimension>*>& packages) {
  mPhysicsPackages = packages;
}

//------------------------------------------------------------------------------
// Test if the given physics package is listed in the integrator.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Integrator<Dimension>::
havePhysicsPackage(const Physics<Dimension>& package) const {
  return count(mPhysicsPackages.begin(), mPhysicsPackages.end(), &package) > 0;
}

//------------------------------------------------------------------------------
// Get the unique set of boundary conditions across all physics packages.
//------------------------------------------------------------------------------
template<typename Dimension>
vector<Boundary<Dimension>*>
Integrator<Dimension>::
uniqueBoundaryConditions() const {

  // The result we're going to build up.
  vector<Boundary<Dimension>*> result;

  // Iterate over each physics package.
  for (ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {

    // Iterate over the boundary conditions associated with this package.
    for (ConstBoundaryIterator boundaryItr = (*physicsItr)->boundaryBegin();
         boundaryItr != (*physicsItr)->boundaryEnd();
         ++boundaryItr) {
      if (find(result.begin(), result.end(), *boundaryItr) == result.end())
        result.push_back(*boundaryItr);
    }

  }

  BEGIN_CONTRACT_SCOPE
  // Ensure that all boundary conditions are included in the result
  for (ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) {
    for (ConstBoundaryIterator boundaryItr = (*physicsItr)->boundaryBegin();
         boundaryItr != (*physicsItr)->boundaryEnd();
         ++boundaryItr) {
      ENSURE(find(result.begin(), result.end(), *boundaryItr) != result.end());
    }
  }

  // Also ensure that there are no duplicates.
  for (ConstBoundaryIterator boundaryItr = result.begin();
       boundaryItr != result.end();
       ++boundaryItr) {
    ENSURE(count(result.begin(), result.end(), *boundaryItr) == 1);
  }
  END_CONTRACT_SCOPE

  // That's it.
  return result;
}

//------------------------------------------------------------------------------
// Set up the ghost nodes for boundary conditions.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::setGhostNodes() {

  // Get that DataBase.
  auto& db = accessDataBase();

  // Get the complete set of unique boundary conditions.
  const auto boundaries = uniqueBoundaryConditions();

  // Remove any old ghost node information from the NodeLists.
  for (auto nodeListItr = db.fluidNodeListBegin(); nodeListItr < db.fluidNodeListEnd(); ++nodeListItr) {
    (*nodeListItr)->numGhostNodes(0);
  }


  // If we're need overlap connectivity, we need to double the kernel extent before setting ghost nodes.
  if (mRequireOverlapConnectivity) {
    for (auto nodeListItr = db.fluidNodeListBegin(); nodeListItr < db.fluidNodeListEnd();  ++nodeListItr) {
      auto& neighbor = (*nodeListItr)->neighbor();
      auto maxeta = 2.0*neighbor.kernelExtent();
      neighbor.kernelExtent(maxeta);
    }
  }

  // Update neighboring
  for (auto nodeListItr = db.fluidNodeListBegin(); nodeListItr < db.fluidNodeListEnd();  ++nodeListItr) {
    auto& neighbor = (*nodeListItr)->neighbor();
    neighbor.updateNodes();
  }

  // Iterate over the boundaries and set their ghost node info.
  for (auto boundaryItr = boundaries.begin(); boundaryItr != boundaries.end(); ++boundaryItr) {
    (*boundaryItr)->setAllGhostNodes(db);
    (*boundaryItr)->finalizeGhostBoundary();
    for (auto nodeListItr = db.fluidNodeListBegin(); nodeListItr < db.fluidNodeListEnd();  ++nodeListItr) {
      (*nodeListItr)->neighbor().updateNodes();
    }
  }

  // If we doubled the kernel extents for overlap connectivity, put 'em back.
  if (mRequireOverlapConnectivity) {
    for (auto nodeListItr = db.fluidNodeListBegin(); nodeListItr < db.fluidNodeListEnd();  ++nodeListItr) {
      auto& neighbor = (*nodeListItr)->neighbor();
      auto maxeta = 0.5*neighbor.kernelExtent();
      neighbor.kernelExtent(maxeta);
      neighbor.updateNodes();
    }
  }

  if (mRequireConnectivity) {

    // Update the connectivity.
    db.updateConnectivityMap(mRequireGhostConnectivity, mRequireOverlapConnectivity, mRequireIntersectionConnectivity);

    // If we're culling ghost nodes, do it now.
    if (mCullGhostNodes and 
        (not this->domainDecompositionIndependent()) and
        (not mRequireGhostConnectivity) and
        (not mRequireOverlapConnectivity)) {
      const auto numNodeLists = db.numNodeLists();
      const auto& cm = db.connectivityMap();

      // First build the set of flags indicating which nodes are used.
      FieldList<Dimension, int> flags = db.newGlobalFieldList(0, "active nodes");
      auto nodeListi = 0;
      for (auto nodeListItr = db.nodeListBegin(); nodeListItr < db.nodeListEnd(); ++nodeListItr, ++nodeListi) {
        const auto& nodeList = **nodeListItr;
        for (auto i = 0u; i != nodeList.numInternalNodes(); ++i) {
          flags(nodeListi, i) = 1;
          const vector<vector<int> >& fullConnectivity = cm.connectivityForNode(&nodeList, i);
          for (auto nodeListj = 0u; nodeListj != fullConnectivity.size(); ++nodeListj) {
            const vector<int>& connectivity = fullConnectivity[nodeListj];
            for (vector<int>::const_iterator jItr = connectivity.begin();
                 jItr != connectivity.end();
                 ++jItr) flags(nodeListj, *jItr) = 1;
          }
        }

        // Ghost nodes that are control nodes for other ghost nodes we're keeping must
        // be kept as well.
        const auto firstGhostNode = nodeList.firstGhostNode();
        for (auto boundaryItr = boundaries.begin(); boundaryItr < boundaries.end(); ++boundaryItr) {
          const auto& boundary = **boundaryItr;
          const auto& controlNodes = boundary.controlNodes(nodeList);
          const auto& ghostNodes = boundary.ghostNodes(nodeList);
          // CHECK(controlNodes.size() == ghostNodes.size());  // Not true if this is a DistributedBoundary!
          for (auto i: controlNodes) {
            if (i >= (int)firstGhostNode) flags(nodeListi, i) = 1;
          }

          // Boundary conditions are allowed to opt out of culling entirely.
          if (not boundary.allowGhostCulling()) {
            for (auto i: ghostNodes) flags(nodeListi, i) = 1;
          }
        }
      }

      // Create the index mapping from old to new node orderings.
      FieldList<Dimension, int> old2newIndexMap = db.newGlobalFieldList(int(0), "index map");
      nodeListi = 0;
      for (auto nodeListItr = db.nodeListBegin(); nodeListItr < db.nodeListEnd(); ++nodeListItr, ++nodeListi) {
        const auto numNodes = (**nodeListItr).numNodes();
        for (auto i = 0u; i != numNodes; ++i) old2newIndexMap(nodeListi, i) = i;
      }

      // Now use these flags to cull the boundary conditions.
      vector<int> numNodesRemoved(numNodeLists, 0);
      for (auto boundaryItr = boundaries.begin(); boundaryItr < boundaries.end(); ++boundaryItr) {
        (*boundaryItr)->cullGhostNodes(flags, old2newIndexMap, numNodesRemoved);
      }

      // Patch up the connectivity map.
      db.patchConnectivityMap(flags, old2newIndexMap);

      // Now the boundary conditions have been updated, so we can go ahead and remove
      // the ghost nodes themselves from the NodeLists.
      nodeListi = 0;
      for (auto nodeListItr = db.nodeListBegin(); nodeListItr < db.nodeListEnd(); ++nodeListItr, ++nodeListi) {
        auto& nodeList = **nodeListItr;
        vector<int> nodesToRemove;
        for (auto i = nodeList.firstGhostNode(); i < nodeList.numNodes(); ++i) {
          if (flags(nodeListi, i) == 0) nodesToRemove.push_back(i);
        }
        nodeList.deleteNodes(nodesToRemove);
        nodeList.neighbor().updateNodes();
      }

      // All nodes should now be labeled as keepers.
      BEGIN_CONTRACT_SCOPE
      {
        for (auto nodeListi = 0; nodeListi < (int)numNodeLists; ++nodeListi) {
          ENSURE(flags[nodeListi]->numElements() == 0 or
                 *min_element(flags[nodeListi]->begin(), flags[nodeListi]->end()) == 1);
        }
      }
      END_CONTRACT_SCOPE

      // The ConnectivityMap should be valid too.
      ENSURE(db.connectivityMap().valid());
    }

  // } else {

  //   // We're not connectivity and don't need ghost nodes, so make sure all 
  //   // boundaries are empty.
  //   const vector<Boundary<Dimension>*> boundaries = uniqueBoundaryConditions();
  //   for (ConstBoundaryIterator boundaryItr = boundaries.begin(); 
  //        boundaryItr != boundaries.end();
  //        ++boundaryItr) (*boundaryItr)->reset(db);
  }
}

//------------------------------------------------------------------------------
// Apply the ghost boundary conditions.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::applyGhostBoundaries(State<Dimension>& state,
                                            StateDerivatives<Dimension>& derivs) {

//   // Start our work timer.
//   typedef Timing::Time Time;
//   const Time start = Timing::currentTime();

  // Get that DataBase.
  DataBase<Dimension>& db = accessDataBase();

  // If we're being rigorous about boundaries, we have to reset the ghost nodes.
  const vector<Boundary<Dimension>*> boundaries = uniqueBoundaryConditions();
  if (mRigorousBoundaries) {
    setGhostNodes();

  } else {

    // If we didn't call setGhostNodes, then make each boundary update it's 
    // ghost node info (position and H).
    for (auto boundaryItr = boundaries.begin(); boundaryItr != boundaries.end(); ++boundaryItr) {
      for (auto nodeListItr = db.nodeListBegin(); nodeListItr != db.nodeListEnd(); ++nodeListItr) {
        (*boundaryItr)->updateGhostNodes(**nodeListItr);
      }
      (*boundaryItr)->finalizeGhostBoundary();
    }
    for (auto nodeListItr = db.fluidNodeListBegin(); nodeListItr != db.fluidNodeListEnd(); ++nodeListItr) {
      (*nodeListItr)->neighbor().updateNodes();
    }
  }

  // Iterate over the physics packages, and have them apply ghost boundaries
  // for their state.
  for (auto physicsItr = physicsPackagesBegin(); physicsItr != physicsPackagesEnd(); ++physicsItr) {
    (*physicsItr)->applyGhostBoundaries(state, derivs);
  }

  //   // Update the work per node fields.
  //   double deltaLocal = Timing::difference(start, Timing::currentTime());
  //   int numLocalNodes = 0;
  //   for (typename DataBase<Dimension>::NodeListIterator nodeListItr = db.nodeListBegin();
  //        nodeListItr != db.nodeListEnd(); 
  //        ++nodeListItr) numLocalNodes += (*nodeListItr)->numInternalNodes();
  //   const double workPerNode = deltaLocal/(numLocalNodes + 1.0e-30);
  //   FieldList<Dimension, Scalar> work = db.globalWork();
  //   work += workPerNode;

  // Finalize the boundaries.
  this->finalizeGhostBoundaries();
}

//------------------------------------------------------------------------------
// Finalize the ghost boundary conditions.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::finalizeGhostBoundaries() {

//   // Start our work timer.
//   typedef Timing::Time Time;
//   const Time start = Timing::currentTime();

  // Get that DataBase.
  //DataBase<Dimension>& db = accessDataBase();

  // If we're being rigorous about boundaries, we have to reset the ghost nodes.
  const vector<Boundary<Dimension>*> boundaries = uniqueBoundaryConditions();
  for (ConstBoundaryIterator boundaryItr = boundaries.begin(); 
       boundaryItr != boundaries.end();
       ++boundaryItr) {
    (*boundaryItr)->finalizeGhostBoundary();
  }

  //   // Update the work per node fields.
  //   double deltaLocal = Timing::difference(start, Timing::currentTime());
  //   int numLocalNodes = 0;
  //   for (typename DataBase<Dimension>::NodeListIterator nodeListItr = db.nodeListBegin();
  //        nodeListItr != db.nodeListEnd(); 
  //        ++nodeListItr) numLocalNodes += (*nodeListItr)->numInternalNodes();
  //   const double workPerNode = deltaLocal/(numLocalNodes + 1.0e-30);
  //   FieldList<Dimension, Scalar> work = db.globalWork();
  //   work += workPerNode;
}

//------------------------------------------------------------------------------
// Identify the internal nodes in violation of the boundary conditions.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::setViolationNodes() {

  // Get that DataBase.
  DataBase<Dimension>& db = accessDataBase();

  // Get the complete set of unique boundary conditions.
  const vector<Boundary<Dimension>*> boundaries = uniqueBoundaryConditions();

  // Have each boundary identify the set of nodes that violate it.
  for (ConstBoundaryIterator boundaryItr = boundaries.begin(); 
       boundaryItr != boundaries.end();
       ++boundaryItr) {
    (*boundaryItr)->setAllViolationNodes(db);
  }

  // Fix neighbor information.
  for (typename DataBase<Dimension>::FluidNodeListIterator nodeListItr = db.fluidNodeListBegin();
       nodeListItr != db.fluidNodeListEnd(); 
       ++nodeListItr) {
    (*nodeListItr)->neighbor().updateNodes();
  }
}

//------------------------------------------------------------------------------
// Enforce the boundary conditions.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::enforceBoundaries(State<Dimension>& state,
                                         StateDerivatives<Dimension>& derivs) {

  // Have each boundary identify the set of nodes in violation.  This also resets
  // the positions and H's of the nodes to be in compliance.
  setViolationNodes();

  // Iterate over the physics packages, and have them apply ghost boundaries
  // for their state.
  for (ConstPackageIterator physicsItr = physicsPackagesBegin();
       physicsItr != physicsPackagesEnd();
       ++physicsItr) (*physicsItr)->enforceBoundaries(state, derivs);
}

//------------------------------------------------------------------------------
// Copy the ghost positions and H tensor from one state to another.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::copyGhostState(const State<Dimension>& state0,
                                      State<Dimension>& state1) const {
  const FieldList<Dimension, Vector> x0 = state0.fields(HydroFieldNames::position, Vector::zero);
  const FieldList<Dimension, SymTensor> H0 = state0.fields(HydroFieldNames::H, SymTensor::zero);
  FieldList<Dimension, Vector> x1 = state1.fields(HydroFieldNames::position, Vector::zero);
  FieldList<Dimension, SymTensor> H1 = state1.fields(HydroFieldNames::H, SymTensor::zero);
  for (GhostNodeIterator<Dimension> itr = x0.ghostNodeBegin();
       itr != x0.ghostNodeEnd();
       ++itr) {
    x1(itr) = x0(itr);
    H1(itr) = H0(itr);
  }
}

//------------------------------------------------------------------------------
// Flag for whether we should try to run in a domain decomposition independent/
// reproducing mode.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Integrator<Dimension>::
domainDecompositionIndependent() const {
  return NodeListRegistrar<Dimension>::instance().domainDecompositionIndependent();
}

template<typename Dimension>
void
Integrator<Dimension>::
domainDecompositionIndependent(const bool x) {
  NodeListRegistrar<Dimension>::instance().domainDecompositionIndependent(x);
  // if (x) mCullGhostNodes = false;
}

//------------------------------------------------------------------------------
// Dump the current state of the Integrator to the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::
dumpState(FileIO& file, const string& pathName) const {

  // file.write(dtMin(), pathName + "/dtMin");
  // file.write(dtMax(), pathName + "/dtMax");
  // file.write(dtGrowth(), pathName + "/dtGrowth");
  file.write(lastDt(), pathName + "/lastDt");
  file.write(currentTime(), pathName + "/currentTime");
  file.write(currentCycle(), pathName + "/currentCycle");
}  

//------------------------------------------------------------------------------
// Restore the state of the NodeList from the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Integrator<Dimension>::
restoreState(const FileIO& file, const string& pathName) {

  // file.read(mDtMin, pathName + "/dtMin");
  // file.read(mDtMax, pathName + "/dtMax");
  // file.read(mDtGrowth, pathName + "/dtGrowth");
  file.read(mLastDt, pathName + "/lastDt");
  file.read(mCurrentTime, pathName + "/currentTime");
  file.read(mCurrentCycle, pathName + "/currentCycle");
}

}
