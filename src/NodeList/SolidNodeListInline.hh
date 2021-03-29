namespace Spheral {

//------------------------------------------------------------------------------
// Access the deviatoric stress field.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
Field<Dimension, typename Dimension::SymTensor>&
SolidNodeList<Dimension>::
deviatoricStress() {
  return mDeviatoricStress;
}

template<typename Dimension>
inline
const Field<Dimension, typename Dimension::SymTensor>&
SolidNodeList<Dimension>::
deviatoricStress() const {
  return mDeviatoricStress;
}

//------------------------------------------------------------------------------
// Access the plastic strain field.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
Field<Dimension, typename Dimension::Scalar>&
SolidNodeList<Dimension>::
plasticStrain() {
  return mPlasticStrain;
}

template<typename Dimension>
inline
const Field<Dimension, typename Dimension::Scalar>&
SolidNodeList<Dimension>::
plasticStrain() const {
  return mPlasticStrain;
}

//------------------------------------------------------------------------------
// Access the plastic strain rate field.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
Field<Dimension, typename Dimension::Scalar>&
SolidNodeList<Dimension>::
plasticStrainRate() {
  return mPlasticStrainRate;
}

template<typename Dimension>
inline
const Field<Dimension, typename Dimension::Scalar>&
SolidNodeList<Dimension>::
plasticStrainRate() const {
  return mPlasticStrainRate;
}

//------------------------------------------------------------------------------
// Access the damage field.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
Field<Dimension, typename Dimension::SymTensor>&
SolidNodeList<Dimension>::
damage() {
  return mDamage;
}

template<typename Dimension>
inline
const Field<Dimension, typename Dimension::SymTensor>&
SolidNodeList<Dimension>::
damage() const {
  return mDamage;
}

//------------------------------------------------------------------------------
// Access the fragment ID field.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
Field<Dimension, int>&
SolidNodeList<Dimension>::
fragmentIDs() {
  return mFragmentIDs;
}

template<typename Dimension>
inline
const Field<Dimension, int>&
SolidNodeList<Dimension>::
fragmentIDs() const {
  return mFragmentIDs;
}

//------------------------------------------------------------------------------
// Access the particle type field.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
Field<Dimension, int>&
SolidNodeList<Dimension>::
particleTypes() {
  return mParticleTypes;
}

template<typename Dimension>
inline
const Field<Dimension, int>&
SolidNodeList<Dimension>::
particleTypes() const {
  return mParticleTypes;
}

//------------------------------------------------------------------------------
// Access the strength model this solid node list is using.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
const StrengthModel<Dimension>&
SolidNodeList<Dimension>::
strengthModel() const {
  return mStrength;
}

}
