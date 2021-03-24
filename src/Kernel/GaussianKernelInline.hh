#include "Geometry/Dimension.hh"
#include "VolumeIntegrationFunctions.hh"
#include "Utilities/DBC.hh"

#include <math.h>

namespace Spheral {

//------------------------------------------------------------------------------
// Construct with the given extent in eta.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
GaussianKernel<Dimension>::GaussianKernel(const double extent):
  Kernel<Dimension, GaussianKernel<Dimension> >() {
  this->setKernelExtent(extent);
  this->setInflectionPoint(sqrt(0.5));
  this->setVolumeNormalization(1.0);
  double volumeIntegral = simpsonsVolumeIntegral<Dimension, GaussianKernel<Dimension> >(*this, 0.0, this->kernelExtent(), 10000);
  CHECK(volumeIntegral > 0.0);
  this->setVolumeNormalization(1.0/volumeIntegral);
}

// template<>
// inline
// GaussianKernel< Dim<1> >::GaussianKernel():
//   Kernel<Dim<1>, GaussianKernel< Dim<1> > >() {
//   setVolumeNormalization(1.0/sqrt(M_PI));
//   setKernelExtent(3.0);
//   setInflectionPoint(sqrt(0.5));
// }

// template<>
// inline
// GaussianKernel< Dim<2> >::GaussianKernel():
//   Kernel<Dim<2>, GaussianKernel< Dim<2> > >() {
//   setVolumeNormalization(1.0/M_PI);
//   setKernelExtent(3.0);
//   setInflectionPoint(sqrt(0.5));
// }

// template<>
// inline
// GaussianKernel< Dim<3> >::GaussianKernel():
//   Kernel<Dim<3>, GaussianKernel< Dim<3> > >() {
//   setVolumeNormalization(1.0/pow(M_PI, 1.5));
//   setKernelExtent(3.0);
//   setInflectionPoint(sqrt(0.5));
// }

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
inline
GaussianKernel<Dimension>::~GaussianKernel() {
}

//------------------------------------------------------------------------------
// Return the kernel weight for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
GaussianKernel<Dimension>::kernelValue(double etaMagnitude, const double Hdet) const {
  CHECK(etaMagnitude >= 0.0);
  CHECK(Hdet >= 0.0);
  return this->volumeNormalization()*Hdet*exp(-etaMagnitude*etaMagnitude);
}

//------------------------------------------------------------------------------
// Return the gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
GaussianKernel<Dimension>::gradValue(double etaMagnitude, const double Hdet) const {
  CHECK(etaMagnitude >= 0.0);
  CHECK(Hdet >= 0.0);
  return -2.0*etaMagnitude*kernelValue(etaMagnitude, Hdet);
}

//------------------------------------------------------------------------------
// Return the second derivative value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
GaussianKernel<Dimension>::grad2Value(double etaMagnitude, const double Hdet) const {
  CHECK(etaMagnitude >= 0.0);
  CHECK(Hdet >= 0.0);
  return 2.0*(2.0*etaMagnitude*etaMagnitude - 1)*kernelValue(etaMagnitude, Hdet);
}

}
