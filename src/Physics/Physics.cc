//---------------------------------Spheral++----------------------------------//
// Physics -- The topmost abstract base class for all physics packages in
// Spheral++.  Physics defines the methods generic to all physics classes,
// but leaves specialized interfaces for particular types of physics (radiative vs.
// hydro, etc.) to a descendent interface class, Physics.
//----------------------------------------------------------------------------//
#include "Physics.hh"
#include "Boundary/Boundary.hh"

using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::abs;

namespace Spheral {

//------------------------------------------------------------------------------
// Default constructor
//------------------------------------------------------------------------------
template<typename Dimension>
Physics<Dimension>::
Physics():
  mBoundaryConditions() {
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
Physics<Dimension>::
~Physics() {
}

//------------------------------------------------------------------------------
// Add a Boundary condition to the end of the current boundary list.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
appendBoundary(Boundary<Dimension>& boundary) {
//   if (!haveBoundary(boundary)) {
    mBoundaryConditions.push_back(&boundary);
//   } else {
//     cerr << "Warning: attempt to append Boundary condition " << &boundary
//          << "to Physics " << this << " which already has it." << endl;
//   }
}

//------------------------------------------------------------------------------
// Add a Boundary condition to the beginning of the current boundary list.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
prependBoundary(Boundary<Dimension>& boundary) {
//   if (!haveBoundary(boundary)) {
    mBoundaryConditions.insert(mBoundaryConditions.begin(), &boundary);
//   } else {
//     cerr << "Warning: attempt to prepend Boundary condition " << &boundary
//          << "to Physics " << this << " which already has it." << endl;
//   }
}

//------------------------------------------------------------------------------
// Clear (erase) the boundary condition list.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
clearBoundaries() {
  mBoundaryConditions = vector<Boundary<Dimension>*>();
}

//------------------------------------------------------------------------------
// Test if the given Boundary condition is listed in the physics package.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Physics<Dimension>::
haveBoundary(const Boundary<Dimension>& boundary) const {
  return std::count(mBoundaryConditions.begin(), mBoundaryConditions.end(), &boundary) > 0;
}

//------------------------------------------------------------------------------
// Provide a default no-op ghost boundary method.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
applyGhostBoundaries(State<Dimension>& /*state*/,
                     StateDerivatives<Dimension>& /*derivs*/) {
}

//------------------------------------------------------------------------------
// Provide a default no-op enforce boundary method.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
enforceBoundaries(State<Dimension>& /*state*/,
                  StateDerivatives<Dimension>& /*derivs*/) {
}

//------------------------------------------------------------------------------
// Provide a default no-op problem startup initialization method.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
initializeProblemStartup(DataBase<Dimension>& /*dataBase*/) {
}

//------------------------------------------------------------------------------
// Provide a default no-op pre-step initialization method.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
preStepInitialize(const DataBase<Dimension>& /*dataBase*/, 
                  State<Dimension>& /*state*/,
                  StateDerivatives<Dimension>& /*derivs*/) {
}

//------------------------------------------------------------------------------
// Provide a default no-op initialization method.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
initialize(const typename Dimension::Scalar /*time*/,
           const typename Dimension::Scalar /*dt*/,
           const DataBase<Dimension>& /*dataBase*/,
           State<Dimension>& /*state*/,
           StateDerivatives<Dimension>& /*derivs*/) {
}

//------------------------------------------------------------------------------
// Provide a default no-op finalization method.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
finalize(const typename Dimension::Scalar /*time*/,
         const typename Dimension::Scalar /*dt*/,
         DataBase<Dimension>& /*dataBase*/,
         State<Dimension>& /*state*/,
         StateDerivatives<Dimension>& /*derivs*/) {
}

//------------------------------------------------------------------------------
// Provide a default no-op finalizeDerivatives method.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
finalizeDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar /*dt*/,
                    const DataBase<Dimension>& /*dataBase*/,
                    const State<Dimension>& /*state*/,
                    StateDerivatives<Dimension>& /*derivs*/) const {
}

//------------------------------------------------------------------------------
// Provide a default no-op postStateUpdate method.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
postStateUpdate(const Scalar /*time*/, 
                const Scalar /*dt*/,
                const DataBase<Dimension>& /*dataBase*/, 
                State<Dimension>& /*state*/,
                StateDerivatives<Dimension>& /*derivatives*/) {
}

//------------------------------------------------------------------------------
// By default assume connectivity needs to be constructed.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Physics<Dimension>::
requireConnectivity() const {
  return true;
}

//------------------------------------------------------------------------------
// By default assume ghost connectivity is not needed.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Physics<Dimension>::
requireGhostConnectivity() const {
  return false;
}

//------------------------------------------------------------------------------
// By default assume overlap connectivity is not needed.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Physics<Dimension>::
requireOverlapConnectivity() const {
  return false;
}

//------------------------------------------------------------------------------
// By default assume intersect connectivity is not needed.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Physics<Dimension>::
requireIntersectionConnectivity() const {
  return false;
}

//------------------------------------------------------------------------------
// By default assume reproducing kernels are not needed.
//------------------------------------------------------------------------------
template<typename Dimension>
std::set<RKOrder>
Physics<Dimension>::
requireReproducingKernels() const {
  return std::set<RKOrder>();
}

//------------------------------------------------------------------------------
// By default assume reproducing kernels second derivatives are not needed.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Physics<Dimension>::
requireReproducingKernelHessian() const {
  return false;
}

//------------------------------------------------------------------------------
// By default assume reproducing kernel correction in finalize is not needed.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Physics<Dimension>::
updateReproducingKernelsInFinalize() const {
  return false;
}

//------------------------------------------------------------------------------
// Provide a default method for the extraEnergy method, which will return 0.0
// for classes that don't have their own energy.
//------------------------------------------------------------------------------
template<typename Dimension>
typename Dimension::Scalar
Physics<Dimension>::
extraEnergy() const {
  return 0.0;
}

//------------------------------------------------------------------------------
// Provide a default method for the extraMomentum method, which will return 
// the zero vector for classes that don't have their own momentum.
//------------------------------------------------------------------------------
template<typename Dimension>
typename Dimension::Vector
Physics<Dimension>::
extraMomentum() const {
  return typename Dimension::Vector();
}

//------------------------------------------------------------------------------
// Defaul noop for extra viz state.
//------------------------------------------------------------------------------
template<typename Dimension>
void
Physics<Dimension>::
registerAdditionalVisualizationState(DataBase<Dimension>& /*dataBase*/,
                                     State<Dimension>& /*state*/) {
}

}
