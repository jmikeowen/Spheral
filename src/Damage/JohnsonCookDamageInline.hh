namespace Spheral {

//------------------------------------------------------------------------------
// Access the attributes
//------------------------------------------------------------------------------
template<typename Dimension>
const SolidNodeList<Dimension>&
JohnsonCookDamage<Dimension>::
nodeList() const {
  return mNodeList;
}

template<typename Dimension>
const Field<Dimension, typename Dimension::Scalar>&
JohnsonCookDamage<Dimension>::
failureStrain() const {
  return mFailureStrain;
}

template<typename Dimension>
const Field<Dimension, typename Dimension::Scalar>&
JohnsonCookDamage<Dimension>::
meltSpecificEnergy() const {
  return mMeltSpecificEnergy;
}

template<typename Dimension>
const Field<Dimension, typename Dimension::Scalar>&
JohnsonCookDamage<Dimension>::
D1() const {
  return mD1;
}

template<typename Dimension>
const Field<Dimension, typename Dimension::Scalar>&
JohnsonCookDamage<Dimension>::
D2() const {
  return mD2;
}

template<typename Dimension>
double
JohnsonCookDamage<Dimension>::
D3() const {
  return mD3;
}

template<typename Dimension>
double
JohnsonCookDamage<Dimension>::
D4() const {
  return mD4;
}

template<typename Dimension>
double
JohnsonCookDamage<Dimension>::
D5() const {
  return mD5;
}

template<typename Dimension>
double
JohnsonCookDamage<Dimension>::
epsilondot0() const {
  return mepsilondot0;
}

template<typename Dimension>
double
JohnsonCookDamage<Dimension>::
Tcrit() const {
  return mTcrit;
}

template<typename Dimension>
double
JohnsonCookDamage<Dimension>::
sigmamax() const {
  return msigmamax;
}

template<typename Dimension>
double
JohnsonCookDamage<Dimension>::
efailmin() const {
  return mefailmin;
}

}
