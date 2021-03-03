//---------------------------------Spheral++----------------------------------//
// SolidFieldNames -- A collection of standard Field names for solid materials.
//
// Created by JMO, Wed Sep 8 11:05:49 2004
//----------------------------------------------------------------------------//
#ifndef _Spheral_SolidFieldNames_
#define _Spheral_SolidFieldNames_

#include <string>

namespace Spheral {

struct SolidFieldNames {
  static const std::string deviatoricStress;
  static const std::string deviatoricStressTT;
  static const std::string plasticStrain;
  static const std::string plasticStrainRate;
  static const std::string scalarDamage;
  static const std::string tensorDamage;
  static const std::string damageCoupling;
  static const std::string strain;
  static const std::string strainTensor;
  static const std::string effectiveStrainTensor;
  static const std::string bulkModulus;
  static const std::string shearModulus;
  static const std::string YoungsModulus;
  static const std::string longitudinalSoundSpeed;
  static const std::string yieldStrength;
  static const std::string flaws;
  static const std::string effectiveFlaws;
  static const std::string porosityAlpha;
  static const std::string porosityStrain;
  static const std::string porosityAlpha0;
  static const std::string porosityc0;
  static const std::string fragmentIDs;
  static const std::string particleTypes;
  static const std::string meltSpecificEnergy;
};

}

#else

namespace Spheral {
  struct SolidFieldNames;
}

#endif
