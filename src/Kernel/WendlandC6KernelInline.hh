#include "Geometry/Dimension.hh"

namespace Spheral {
namespace KernelSpace {

//------------------------------------------------------------------------------
// Empty constructor
//------------------------------------------------------------------------------
template<>
inline
WendlandC6Kernel< Dim<1> >::WendlandC6Kernel():
  Kernel<Dim<1>, WendlandC6Kernel< Dim<1> > >() {
  setVolumeNormalization(15.0/8.0);
  setKernelExtent(1.0);
  setInflectionPoint(1.0/3.0);
}

template<>
inline
WendlandC6Kernel< Dim<2> >::WendlandC6Kernel():
  Kernel<Dim<2>, WendlandC6Kernel< Dim<2> > >() {
  setVolumeNormalization(78.0/(14.0*M_PI));
  setKernelExtent(1.0);
  setInflectionPoint(1.0/3.0);
}

template<>
inline
WendlandC6Kernel< Dim<3> >::WendlandC6Kernel():
  Kernel<Dim<3>, WendlandC6Kernel< Dim<3> > >() {
  setVolumeNormalization(1365.0/(128.0*M_PI));
  setKernelExtent(1.0);
  setInflectionPoint(1.0/3.0);
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
inline
WendlandC6Kernel<Dimension>::~WendlandC6Kernel() {
}

//------------------------------------------------------------------------------
// Return the kernel weight for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
WendlandC6Kernel<Dimension>::kernelValue(double etaMagnitude, double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);


    double eta2 = etaMagnitude*etaMagnitude;
    return this->volumeNormalization()*Hdet*(pow(1.0-etaMagnitude,8)*
                                             (1.0+8.0*etaMagnitude+25.0*eta2+32.0*etaMagnitude*eta2));

}

//------------------------------------------------------------------------------
// Return the gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
WendlandC6Kernel<Dimension>::gradValue(double etaMagnitude, double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);


    double eta2 = etaMagnitude*etaMagnitude;
    return  this->volumeNormalization()*Hdet*(22.0*pow(etaMagnitude-1.0,7)*
                                              etaMagnitude*(16.0*eta2+7.0*etaMagnitude+1.0));

}

//------------------------------------------------------------------------------
// Return the second derivative value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
WendlandC6Kernel<Dimension>::grad2Value(double etaMagnitude, double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);


    const double eta2 = etaMagnitude*etaMagnitude;
    return this->volumeNormalization()*Hdet*(22.0*pow(etaMagnitude-1.0,6)*
                                               (160.0*eta2*etaMagnitude+15.0*eta2-6.0*etaMagnitude-1.0));

}

}
}
