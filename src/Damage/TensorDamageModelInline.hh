namespace Spheral {

//------------------------------------------------------------------------------
// Access the state fields.
//------------------------------------------------------------------------------
template<typename Dimension>
const Field<Dimension, typename Dimension::SymTensor>&
TensorDamageModel<Dimension>::
strain() const {
  return mStrain;
}

template<typename Dimension>
const Field<Dimension, typename Dimension::SymTensor>&
TensorDamageModel<Dimension>::
effectiveStrain() const {
  return mEffectiveStrain;
}

template<typename Dimension>
const Field<Dimension, typename Dimension::Scalar>&
TensorDamageModel<Dimension>::
DdamageDt() const {
  return mDdamageDt;
}

//------------------------------------------------------------------------------
// The strain update algorithm.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
TensorStrainAlgorithm
TensorDamageModel<Dimension>::
strainAlgorithm() const {
  return mStrainAlgorithm;
}

//------------------------------------------------------------------------------
// Flag to determine if damage in compression is allowed.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
bool
TensorDamageModel<Dimension>::
damageInCompression() const {
  return mDamageInCompression;
}

template<typename Dimension>
inline
void
TensorDamageModel<Dimension>::
damageInCompression(bool x) {
  mDamageInCompression = x;
}

//------------------------------------------------------------------------------
// Threshold for the damage beyond which a node no longer votes on the time
// step.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
TensorDamageModel<Dimension>::
criticalDamageThreshold() const {
  return mCriticalDamageThreshold;
}

template<typename Dimension>
inline
void
TensorDamageModel<Dimension>::
criticalDamageThreshold(double x) {
  mCriticalDamageThreshold = x;
}

}
