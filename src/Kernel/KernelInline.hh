#include <math.h>
#include "Utilities/DBC.hh"
#include "Geometry/GeomVector.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Return as a reference descendent class.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
Descendant&
Kernel<Dimension, Descendant>::asDescendant() const {
  return static_cast<Descendant&>(const_cast<Kernel<Dimension, Descendant>&>(*this));
//   return static_cast<Descendant&>(*this);
}

//------------------------------------------------------------------------------
// Empty constructor
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
Kernel<Dimension, Descendant>::Kernel():
  mVolumeNormalization(0.0),
  mKernelExtent(0.0),
  mInflectionPoint(0.0) {
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
Kernel<Dimension, Descendant>::~Kernel() {
}

//------------------------------------------------------------------------------
// Assignment
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
Kernel<Dimension, Descendant>&
Kernel<Dimension, Descendant>::operator=(const Kernel<Dimension, Descendant>& rhs) {
  if (this != &rhs) {
    mVolumeNormalization = rhs.volumeNormalization();
    mKernelExtent = rhs.kernelExtent();
    mInflectionPoint = rhs.inflectionPoint();
  }
  return *this;
}

//------------------------------------------------------------------------------
// Return the kernel weight for a given normalized distance or position.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::operator()(const typename Dimension::Vector& eta, 
                                          const typename Dimension::SymTensor& H) const {
  REQUIRE(valid());
  return kernelValue(eta.magnitude(), H.Determinant());
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::operator()(const typename Dimension::Vector& eta, 
                                          const typename Dimension::Scalar& Hdet) const {
  REQUIRE(valid());
  return kernelValue(eta.magnitude(), Hdet);
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::operator()(const double& etaMagnitude, 
                                          const typename Dimension::SymTensor& H) const {
  REQUIRE(valid());
  REQUIRE(etaMagnitude >= 0.0);
  return kernelValue(etaMagnitude, H.Determinant());
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::operator()(const double& etaMagnitude, 
                                          const typename Dimension::Scalar& Hdet) const {
  REQUIRE(valid());
  REQUIRE(etaMagnitude >= 0.0);
  return kernelValue(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// Return the gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad(const typename Dimension::Vector& eta,
                                    const typename Dimension::SymTensor& H) const {
  REQUIRE(valid());
  return gradValue(eta.magnitude(), H.Determinant());
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad(const typename Dimension::Vector& eta,
                                    const typename Dimension::Scalar& Hdet) const {
  REQUIRE(valid());
  return gradValue(eta.magnitude(), Hdet);
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad(const double& etaMagnitude,
                                    const typename Dimension::SymTensor& H) const {
  REQUIRE(valid());
  REQUIRE(etaMagnitude >= 0.0);
  return gradValue(etaMagnitude, H.Determinant());
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad(const double& etaMagnitude,
                                    const typename Dimension::Scalar& Hdet) const {
  REQUIRE(valid());
  REQUIRE(etaMagnitude >= 0.0);
  return gradValue(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// Return the second derivative value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad2(const typename Dimension::Vector& eta,
                                     const typename Dimension::SymTensor& H) const {
  REQUIRE(valid());
  return grad2Value(eta.magnitude(), H.Determinant());
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad2(const typename Dimension::Vector& eta,
                                     const typename Dimension::Scalar& Hdet) const {
  REQUIRE(valid());
  return grad2Value(eta.magnitude(), Hdet);
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad2(const double& etaMagnitude,
                                     const typename Dimension::SymTensor& H) const {
  REQUIRE(valid());
  REQUIRE(etaMagnitude >= 0.0);
  return grad2Value(etaMagnitude, H.Determinant());
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad2(const double& etaMagnitude,
                                     const typename Dimension::Scalar& Hdet) const {
  REQUIRE(valid());
  REQUIRE(etaMagnitude >= 0.0);
  return grad2Value(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// Return the gradient with respect to h value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::gradh(const typename Dimension::Vector& eta,
                                     const typename Dimension::SymTensor& H) const {
  REQUIRE(valid());
  return gradhValue(eta.magnitude(), H.Determinant());
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::gradh(const typename Dimension::Vector& eta,
                                     const typename Dimension::Scalar& Hdet) const {
  REQUIRE(valid());
  return gradhValue(eta.magnitude(), Hdet);
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::gradh(const double& etaMagnitude,
                                     const typename Dimension::SymTensor& H) const {
  REQUIRE(valid());
  REQUIRE(etaMagnitude >= 0.0);
  return gradhValue(etaMagnitude, H.Determinant());
}

template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::gradh(const double& etaMagnitude,
                                     const typename Dimension::Scalar& Hdet) const {
  REQUIRE(valid());
  REQUIRE(etaMagnitude >= 0.0);
  return gradhValue(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// All kernels must redefine this method to give the W value for a given
// distance and H determinant.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::kernelValue(double etaMagnitude, const double Hdet) const {
  REQUIRE(valid());
  return asDescendant().kernelValue(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// All kernels must redefine this method to give the gradient value for a given
// distance and H determinant.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::gradValue(double etaMagnitude, const double Hdet) const {
  REQUIRE(valid());
  return asDescendant().gradValue(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// All kernels must redefine this method to give the second derivative value for
// a given distance and H determinant.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::grad2Value(double etaMagnitude, const double Hdet) const {
  REQUIRE(valid());
  return asDescendant().grad2Value(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// Compute the gradient with respect to h.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::gradhValue(double etaMagnitude, const double Hdet) const {
  REQUIRE(valid());
  return -etaMagnitude * Dimension::rootnu(Hdet) * asDescendant().gradValue(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// Return the volume normalization.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::volumeNormalization() const {
  return mVolumeNormalization;
}

//------------------------------------------------------------------------------
// Set the volume normalization.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
void
Kernel<Dimension, Descendant>::setVolumeNormalization(double volumeNormalization) {
  REQUIRE(volumeNormalization > 0.0);
  mVolumeNormalization = volumeNormalization;
}

//------------------------------------------------------------------------------
// Return the kernel extent.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::kernelExtent() const {
  return mKernelExtent;
}

//------------------------------------------------------------------------------
// Set the kernel extent.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
void
Kernel<Dimension, Descendant>::setKernelExtent(double extent) {
  REQUIRE(extent > 0.0);
  mKernelExtent = extent;
}

//------------------------------------------------------------------------------
// Return the inflection point.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
double
Kernel<Dimension, Descendant>::inflectionPoint() const {
  return mInflectionPoint;
}

//------------------------------------------------------------------------------
// Set the inflection point.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
void
Kernel<Dimension, Descendant>::setInflectionPoint(double x) {
  mInflectionPoint = x;
}

//------------------------------------------------------------------------------
// Test if the kernel is valid.
//------------------------------------------------------------------------------
template<typename Dimension, typename Descendant>
inline
bool
Kernel<Dimension, Descendant>::valid() const {
  return (volumeNormalization() > 0.0 && kernelExtent() > 0.0);
}

}
