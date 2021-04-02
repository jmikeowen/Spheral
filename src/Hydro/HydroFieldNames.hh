//---------------------------------Spheral++----------------------------------//
// HydroFieldNames -- A collection of standard Field names for the hydro 
// physics package.
//
// Created by JMO, Sat Aug 28 21:16:14 2004
//----------------------------------------------------------------------------//
#ifndef _Spheral_HydroFieldNames_
#define _Spheral_HydroFieldNames_

#include "Utilities/SPHString.hh"

#include <string>

namespace Spheral {

struct HydroFieldNames {
  static const SPHString mass;
  static const SPHString position;
  static const SPHString velocity;
  static const SPHString H;
  static const SPHString work;
  static const SPHString velocityGradient;
  static const SPHString internalVelocityGradient;
  static const SPHString hydroAcceleration;
  static const SPHString massDensity;
  static const SPHString normalization;
  static const SPHString specificThermalEnergy;
  static const SPHString maxViscousPressure;
  static const SPHString effectiveViscousPressure;
  static const SPHString massDensityCorrection;
  static const SPHString viscousWork;
  static const SPHString XSPHDeltaV;
  static const SPHString XSPHWeightSum;
  static const SPHString Hsmooth;
  static const SPHString massFirstMoment;
  static const SPHString massSecondMoment;
  static const SPHString weightedNeighborSum;
  static const SPHString pressure;
  static const SPHString temperature;
  static const SPHString soundSpeed;
  static const SPHString pairAccelerations;
  static const SPHString pairWork;
  static const SPHString gamma;
  static const SPHString entropy;
  static const SPHString PSPHcorrection;
  static const SPHString omegaGradh;
  static const SPHString numberDensitySum;
  static const SPHString timeStepMask;
  static const SPHString surfacePoint;
  static const SPHString voidPoint;
  static const SPHString etaVoidPoints;
  static const SPHString cells;
  static const SPHString cellFaceFlags;
  static const SPHString M_SPHCorrection;
  static const SPHString volume;
  static const SPHString linearMomentum;
  static const SPHString totalEnergy;
  static const SPHString mesh;
  static const SPHString hourglassMask;
  static const SPHString faceVelocity;
  static const SPHString faceForce;
  static const SPHString faceMass;
  static const SPHString polyvols;
  static const SPHString massDensityGradient;
  static const SPHString ArtificialViscousClMultiplier;
  static const SPHString ArtificialViscousCqMultiplier;
  static const SPHString specificHeat;
  static const SPHString normal;
  static const SPHString surfaceArea;
};

}

#else

namespace Spheral {
  struct HydroFieldNames;
}

#endif
