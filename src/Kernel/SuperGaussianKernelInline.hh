#include "Geometry/Dimension.hh"

#include <math.h>

namespace Spheral {

//------------------------------------------------------------------------------
// Empty constructor
//------------------------------------------------------------------------------
// template<typename Dimension>
// inline
// SuperGaussianKernel<Dimension>::SuperGaussianKernel():
//   Kernel<Dimension, SuperGaussianKernel<Dimension> >() {
// }

template<>
inline
SuperGaussianKernel< Dim<1> >::SuperGaussianKernel():
  Kernel<Dim<1>, SuperGaussianKernel< Dim<1> > >() {
  setVolumeNormalization(1.0/std::sqrt(M_PI));
  setKernelExtent(3.0);
}

template<>
inline
SuperGaussianKernel< Dim<2> >::SuperGaussianKernel():
  Kernel<Dim<2>, SuperGaussianKernel< Dim<2> > >() {
  setVolumeNormalization(1.0/M_PI);
  setKernelExtent(3.0);
}

template<>
inline
SuperGaussianKernel< Dim<3> >::SuperGaussianKernel():
  Kernel<Dim<3>, SuperGaussianKernel< Dim<3> > >() {
  setVolumeNormalization(1.0/std::pow(M_PI, 1.5));
  setKernelExtent(3.0);
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
inline
SuperGaussianKernel<Dimension>::~SuperGaussianKernel() {
}

//------------------------------------------------------------------------------
// Return the kernel weight for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
SuperGaussianKernel<Dimension>::kernelValue(double etaMagnitude, const double Hdet) const {
  CHECK(etaMagnitude >= 0.0);
  CHECK(Hdet >= 0.0);
  double etaMagnitude2 = etaMagnitude*etaMagnitude;
  return this->volumeNormalization()*Hdet*(mKW - etaMagnitude2)*exp(-etaMagnitude2);
}

//------------------------------------------------------------------------------
// Return the gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
SuperGaussianKernel<Dimension>::gradValue(double etaMagnitude, const double Hdet) const {
  CHECK(etaMagnitude >= 0.0);
  CHECK(Hdet >= 0.0);
  double etaMagnitude2 = etaMagnitude*etaMagnitude;
  return -2.0*this->volumeNormalization()*Hdet*
    etaMagnitude*(mKW + 1.0 - etaMagnitude2)*exp(-etaMagnitude2);
}

//------------------------------------------------------------------------------
// Return the second derivative value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
SuperGaussianKernel<Dimension>::grad2Value(double etaMagnitude, const double Hdet) const {
  CHECK(etaMagnitude >= 0.0);
  CHECK(Hdet >= 0.0);
  double etaMagnitude2 = etaMagnitude*etaMagnitude;
  return 2.0*this->volumeNormalization()*Hdet*
    ((2.0*mKW + 5.0)*etaMagnitude2 - 4.0*etaMagnitude2*etaMagnitude2 - mKW - 1.0)*
    exp(-etaMagnitude2);
}

}
