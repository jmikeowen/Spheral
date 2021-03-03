//---------------------------------Spheral++----------------------------------//
// Verlet -- Advance the set of Physics packages in time using the second
// order Verlet algorithm.  This method is symplectic in the absence of 
// dissipation.
//
// Based on the description in 
// Monaghan JJ. Smoothed particle hydrodynamics. Reports on progress in physics. 2005
//
// Created by JMO, Sat Aug 23 10:33:33 PDT 2014
//----------------------------------------------------------------------------//
#include "DataOutput/Restart.hh"
#include "Verlet.hh"
#include "DataBase/DataBase.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/IncrementFieldList.hh"
#include "Field/FieldList.hh"
#include "Field/NodeIterators.hh"
#include "Physics/Physics.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Utilities/DBC.hh"
#include "Utilities/Timer.hh"

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

// Declare the timers
extern Timer TIME_Verlet;
extern Timer TIME_VerletPreInit;
extern Timer TIME_VerletCopyPos0;
extern Timer TIME_VerletDt;
extern Timer TIME_VerletCopyState0;
extern Timer TIME_VerletEvalDerivs1;
extern Timer TIME_VerletPredict1;
extern Timer TIME_VerletDtCheck;
extern Timer TIME_VerletMidPointCopy;
extern Timer TIME_VerletPredict2;
extern Timer TIME_VerletEvalDerivs2;
extern Timer TIME_VerletUpdateState;
extern Timer TIME_VerletFinalize;

namespace Spheral {

//------------------------------------------------------------------------------
// Empty constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
Verlet<Dimension>::Verlet():
  Integrator<Dimension>() {
}

//------------------------------------------------------------------------------
// Construct with the given DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
Verlet<Dimension>::
Verlet(DataBase<Dimension>& dataBase):
  Integrator<Dimension>(dataBase) {
}

//------------------------------------------------------------------------------
// Construct with the given DataBase and Physics packages.
//------------------------------------------------------------------------------
template<typename Dimension>
Verlet<Dimension>::
Verlet(DataBase<Dimension>& dataBase,
               const vector<Physics<Dimension>*>& physicsPackages):
  Integrator<Dimension>(dataBase, physicsPackages) {
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
Verlet<Dimension>::~Verlet() {
}

//------------------------------------------------------------------------------
// Assignment
//------------------------------------------------------------------------------
template<typename Dimension>
Verlet<Dimension>&
Verlet<Dimension>::
operator=(const Verlet<Dimension>& rhs) {
  if (this != &rhs) {
    Integrator<Dimension>::operator=(rhs);
  }
  return *this;
}

//------------------------------------------------------------------------------
// Take a step.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
Verlet<Dimension>::
step(typename Dimension::Scalar maxTime,
     State<Dimension>& state,
     StateDerivatives<Dimension>& derivs) {

  TIME_Verlet.start();

  // Get the current time and data base.
  auto  t = this->currentTime();
  auto& db = this->accessDataBase();

  // Initalize the integrator.
  TIME_VerletPreInit.start();
  this->preStepInitialize(state, derivs);
  TIME_VerletPreInit.stop();

  // Copy the beginning of step positions.
  TIME_VerletCopyPos0.start();
  auto pos0 = state.fields(HydroFieldNames::position, Vector::zero);
  pos0.copyFields();
  TIME_VerletCopyPos0.stop();

  // Determine the minimum timestep across all packages.
  TIME_VerletDt.start();
  const Scalar dtMin = min(this->dtMin(), maxTime - t);
  const Scalar dtMax = min(this->dtMax(), maxTime - t);
  const Scalar dt0 = this->selectDt(dtMin, dtMax, state, derivs);
  const Scalar hdt0 = 0.5*dt0;
  const auto dtcheck = this->allowDtCheck();
  const auto dtcheckFrac = this->dtCheckFrac();
  TIME_VerletDt.start();

  // If we're doing dt checking, we need to copy the initial state.
  State<Dimension> state0;
  if (dtcheck) {
    TIME_VerletCopyState0.start();
    state0 = state;
    state0.copyState();
    TIME_VerletCopyState0.stop();
  }

  // Evaluate the beginning of step derivatives.
  TIME_VerletEvalDerivs1.start();
  this->initializeDerivatives(t, dt0, state, derivs);
  derivs.Zero();
  this->evaluateDerivatives(t, dt0, db, state, derivs);
  this->finalizeDerivatives(t, dt0, db, state, derivs);
  TIME_VerletEvalDerivs1.stop();

  // Predict state at the mid-point.
  // state.timeAdvanceOnly(true);
  TIME_VerletPredict1.start();
  state.update(derivs, hdt0, t, dt0);
  this->enforceBoundaries(state, derivs);
  this->applyGhostBoundaries(state, derivs);
  this->postStateUpdate(t + hdt0, hdt0, db, state, derivs);
  this->finalizeGhostBoundaries();
  TIME_VerletPredict1.stop();

  // Check if the timestep is still a good idea...
  if (dtcheck) {
    TIME_VerletDtCheck.start();
    const auto dtnew = this->selectDt(dtMin,
                                      dtMax,
                                      state,
                                      derivs);
    if (dtnew < dtcheckFrac*dt0) {
      this->currentTime(t);
      state.assign(state0);
      return false;
      TIME_VerletDtCheck.stop();
    }
    TIME_VerletDtCheck.stop();
  }

  // Copy the mid-point state.
  TIME_VerletMidPointCopy.start();
  State<Dimension> state12(state);
  state12.copyState();
  TIME_VerletMidPointCopy.stop();

  // Advance the position to the end of step using the half-step velocity.
  TIME_VerletPredict2.start();
  auto vel12 = state.fields(HydroFieldNames::velocity, Vector::zero);
  pos0 += dt0*vel12;

  // Predict state at the end-point, but override the positions with our time-centered prediction.
  state.update(derivs, hdt0, t, dt0);
  {
    auto pos = state.fields(HydroFieldNames::position, Vector::zero);
    pos.assignFields(pos0);
  }
  this->enforceBoundaries(state, derivs);
  this->applyGhostBoundaries(state, derivs);
  this->postStateUpdate(t + dt0, dt0, db, state, derivs);
  this->finalizeGhostBoundaries();
  TIME_VerletPredict2.stop();

  // Evaluate the derivatives at the predicted end-point.
  TIME_VerletEvalDerivs2.start();
  this->currentTime(t + dt0);
  this->initializeDerivatives(t + dt0, dt0, state, derivs);
  derivs.Zero();
  this->evaluateDerivatives(t + dt0, dt0, db, state, derivs);
  this->finalizeDerivatives(t + dt0, dt0, db, state, derivs);
  TIME_VerletEvalDerivs2.stop();

  // Check if the timestep is still a good idea...
  if (dtcheck) {
    TIME_VerletDtCheck.start();
    const auto dtnew = this->selectDt(dtMin,
                                      dtMax,
                                      state,
                                      derivs);
    if (dtnew < dtcheckFrac*dt0) {
      this->currentTime(t);
      state.assign(state0);
      TIME_VerletDtCheck.stop();
      return false;
    }
    TIME_VerletDtCheck.stop();
  }

  // Correct the final state by the end-point derivatives.
  TIME_VerletUpdateState.start();
  state.assign(state12);
  // state.timeAdvanceOnly(false);
  state.update(derivs, hdt0, t + hdt0, dt0);
  {
    auto pos = state.fields(HydroFieldNames::position, Vector::zero);
    pos.assignFields(pos0);
  }
  this->enforceBoundaries(state, derivs);
  this->applyGhostBoundaries(state, derivs);
  this->postStateUpdate(t + dt0, dt0, db, state, derivs);
  this->finalizeGhostBoundaries();
  TIME_VerletUpdateState.stop();

  // Apply any physics specific finalizations.
  TIME_VerletFinalize.start();
  this->postStepFinalize(t + dt0, dt0, state, derivs);

  // Set the new current time and last time step.
  this->currentCycle(this->currentCycle() + 1);
  this->lastDt(dt0);
  TIME_VerletFinalize.stop();
  TIME_Verlet.stop();

  return true;
}

}
