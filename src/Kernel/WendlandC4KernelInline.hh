#include "Geometry/Dimension.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Empty constructor
//------------------------------------------------------------------------------
template<>
inline
WendlandC4Kernel< Dim<1> >::WendlandC4Kernel():
  Kernel<Dim<1>, WendlandC4Kernel< Dim<1> > >() {
  setVolumeNormalization(3.0/2.0);
  setKernelExtent(1.0);
  setInflectionPoint(0.275978);
}

template<>
inline
WendlandC4Kernel< Dim<2> >::WendlandC4Kernel():
  Kernel<Dim<2>, WendlandC4Kernel< Dim<2> > >() {
  setVolumeNormalization(9.0/(M_PI));
  setKernelExtent(1.0);
  setInflectionPoint(0.235571);
}

template<>
inline
WendlandC4Kernel< Dim<3> >::WendlandC4Kernel():
  Kernel<Dim<3>, WendlandC4Kernel< Dim<3> > >() {
  setVolumeNormalization(495.0/(32.0*M_PI));
  setKernelExtent(1.0);
  setInflectionPoint(0.235571);
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
inline
WendlandC4Kernel<Dimension>::~WendlandC4Kernel() {
}

//------------------------------------------------------------------------------
// Return the kernel weight for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
WendlandC4Kernel<Dimension>::kernelValue(double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);

  double eta2 = etaMagnitude*etaMagnitude;
  if(Dimension::nDim == 1)
      return this->volumeNormalization()*Hdet*(pow(1.0-etaMagnitude,5)*
                                             (1.0+5.0*etaMagnitude+8.0*eta2))*(etaMagnitude < 1.0);
  else
      return this->volumeNormalization()*Hdet*(pow(1.0-etaMagnitude,6)*
                                             (1.0+6.0*etaMagnitude+(35.0/3.0)*eta2))*(etaMagnitude < 1.0);
}

//------------------------------------------------------------------------------
// Return the gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
WendlandC4Kernel<Dimension>::gradValue(double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);

  if(Dimension::nDim == 1)
      return this->volumeNormalization()*Hdet*(-14.0*pow(1.0-etaMagnitude,4)*etaMagnitude*
                                             (1.0+4.0*etaMagnitude))*(etaMagnitude < 1.0);
  else
      return  this->volumeNormalization()*Hdet*((56.0/3.0)*pow(etaMagnitude-1.0,5)*
                                              etaMagnitude*(5.0*etaMagnitude+1.0))*(etaMagnitude < 1.0);


}

//------------------------------------------------------------------------------
// Return the second derivative value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
WendlandC4Kernel<Dimension>::grad2Value(double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);
    const double eta2 = etaMagnitude*etaMagnitude;
    if(Dimension::nDim == 1)
      return this->volumeNormalization()*Hdet*(-14.0*pow(etaMagnitude-1.0,3)*
                                             (24.0*eta2-3.0*etaMagnitude-1.0))*(etaMagnitude < 1.0);
    else
      return this->volumeNormalization()*Hdet*((56.0/3.0)*pow(etaMagnitude-1.0,4)*
                                               (35.0*eta2-4.0*etaMagnitude-1.0))*(etaMagnitude < 1.0);



}

}
