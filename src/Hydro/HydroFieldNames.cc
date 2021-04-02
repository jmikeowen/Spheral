//---------------------------------Spheral++----------------------------------//
// HydroFieldNames -- A collection of standard Field names for the hydro 
// physics package.
//
// Created by JMO, Sat Aug 28 21:16:14 2004
//----------------------------------------------------------------------------//

#include "HydroFieldNames.hh"

const SPHString Spheral::HydroFieldNames::mass = "mass";
const SPHString Spheral::HydroFieldNames::position = "position";
const SPHString Spheral::HydroFieldNames::velocity = "velocity";
const SPHString Spheral::HydroFieldNames::H = "H";
const SPHString Spheral::HydroFieldNames::work = "work";
const SPHString Spheral::HydroFieldNames::velocityGradient = "velocity gradient";
const SPHString Spheral::HydroFieldNames::internalVelocityGradient = "internal velocity gradient";
const SPHString Spheral::HydroFieldNames::hydroAcceleration = "delta " + Spheral::HydroFieldNames::velocity + " hydro";              // Note here we *must* start with "delta " to work with IncrementFieldList!
const SPHString Spheral::HydroFieldNames::massDensity = "mass density";
const SPHString Spheral::HydroFieldNames::normalization = "normalization";
const SPHString Spheral::HydroFieldNames::specificThermalEnergy = "specific thermal energy";
const SPHString Spheral::HydroFieldNames::maxViscousPressure = "max viscous pressure";
const SPHString Spheral::HydroFieldNames::effectiveViscousPressure = "effective viscous pressure";
const SPHString Spheral::HydroFieldNames::massDensityCorrection = "density summation correction";
const SPHString Spheral::HydroFieldNames::viscousWork = "viscous work rate";
const SPHString Spheral::HydroFieldNames::XSPHDeltaV = "XSPH delta vi";
const SPHString Spheral::HydroFieldNames::XSPHWeightSum = "XSPH weight sum";
const SPHString Spheral::HydroFieldNames::Hsmooth = "H smooth";
const SPHString Spheral::HydroFieldNames::massFirstMoment = "mass first moment";
const SPHString Spheral::HydroFieldNames::massSecondMoment = "mass second moment";
const SPHString Spheral::HydroFieldNames::weightedNeighborSum = "weighted neighbor sum";
const SPHString Spheral::HydroFieldNames::pressure = "pressure";
const SPHString Spheral::HydroFieldNames::temperature = "temperature";
const SPHString Spheral::HydroFieldNames::soundSpeed = "sound speed";
const SPHString Spheral::HydroFieldNames::pairAccelerations = "pair-wise accelerations";
const SPHString Spheral::HydroFieldNames::pairWork = "pair-wise work";
const SPHString Spheral::HydroFieldNames::omegaGradh = "grad h corrections";
const SPHString Spheral::HydroFieldNames::gamma = "ratio of specific heats";
const SPHString Spheral::HydroFieldNames::entropy = "entropy";
const SPHString Spheral::HydroFieldNames::PSPHcorrection = "PSPH Correction";
const SPHString Spheral::HydroFieldNames::numberDensitySum = "number density sum";
const SPHString Spheral::HydroFieldNames::timeStepMask = "time step mask";
const SPHString Spheral::HydroFieldNames::surfacePoint = "surface point";
const SPHString Spheral::HydroFieldNames::voidPoint = "void point";
const SPHString Spheral::HydroFieldNames::etaVoidPoints = "eta void points";
const SPHString Spheral::HydroFieldNames::cells = "cells";
const SPHString Spheral::HydroFieldNames::cellFaceFlags = "cell face flags";
const SPHString Spheral::HydroFieldNames::M_SPHCorrection = "M SPH gradient correction";
const SPHString Spheral::HydroFieldNames::volume = "node volume";
const SPHString Spheral::HydroFieldNames::linearMomentum = "linear momentum";
const SPHString Spheral::HydroFieldNames::totalEnergy = "total energy";
const SPHString Spheral::HydroFieldNames::mesh = "mesh";
const SPHString Spheral::HydroFieldNames::hourglassMask = "hourglass mask";
const SPHString Spheral::HydroFieldNames::faceVelocity = "face velocity";
const SPHString Spheral::HydroFieldNames::faceForce = "face force";
const SPHString Spheral::HydroFieldNames::faceMass = "face mass";
const SPHString Spheral::HydroFieldNames::polyvols = "poly faceted volumes";
const SPHString Spheral::HydroFieldNames::massDensityGradient = "mass density gradient";
const SPHString Spheral::HydroFieldNames::ArtificialViscousClMultiplier = "Cl multiplier for artificial viscosity";
const SPHString Spheral::HydroFieldNames::ArtificialViscousCqMultiplier = "Cq multiplier for artificial viscosity";
const SPHString Spheral::HydroFieldNames::specificHeat = "specific heat";
const SPHString Spheral::HydroFieldNames::normal = "outward normal direction";
const SPHString Spheral::HydroFieldNames::surfaceArea = "boundary surface area";
