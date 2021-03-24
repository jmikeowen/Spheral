//---------------------------------Spheral++----------------------------------//
// HKernel -- A kernel modifier for use by the ideal H methods of the NodeList.
//
// Based and generalized from the counting kernel proposed in
// Thacker, Tittley, Pearce, Couchman, & Thomas 2000, MNRAS, 319, 619.
//
// Volume normalizations:
// 1-D:  A = 1.0
// 2-D:  A = 1.0
// 3-D:  A = 1.0
//
// The idea is to take an existing kernel which is going to be used in a 
// computation, and give a modified version as:
//               { 1.0,   for eta \in [0.0, 0.75*kernelExtent]
// WH(W, eta) = {
//               { W(eta)/(W(0.75), for eta \in [0.75*kernelExtent, kernelExtent]
//
// Created by J. Michael Owen, Thu Apr 22 16:00:39 2004
//---------------------------------Spheral++----------------------------------//
#include "HKernel.hh"
#include "VolumeIntegrationFunctions.hh"

#include <math.h>

namespace Spheral {


//------------------------------------------------------------------------------
// Constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
HKernel<Dimension>::
HKernel(double extent):
  Kernel<Dimension, HKernel<Dimension> >(),
  mInterval14(0.0),
  mInterval34(0.0),
  mPeriod(0.0),
  mOffset(0.5*M_PI),
  mNs(0.0) {
  this->setVolumeNormalization(1.0);
  this->setInflectionPoint(0.0);
  this->matchKernelExtent(extent);
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template<typename Dimension>
HKernel<Dimension>::
~HKernel() {
}

//------------------------------------------------------------------------------
// Set the kernel extent and tail-off to match the given kernel.
//------------------------------------------------------------------------------
template<typename Dimension>
void
HKernel<Dimension>::
matchKernel(const TableKernel<Dimension>& W) {
  matchKernelExtent(W.kernelExtent());
}

//------------------------------------------------------------------------------
// Set the kernel extent and tail-off to match the given kernel.
//------------------------------------------------------------------------------
template<typename Dimension>
void
HKernel<Dimension>::
matchKernelExtent(double extent) {
  REQUIRE(extent > 0.0);
  if (this->kernelExtent() != extent) {
    this->setKernelExtent(extent);
    mInterval14 = 0.25*extent;
    mInterval34 = 0.75*extent;
    mPeriod = 4.0*M_PI/extent;
    mNs = simpsonsVolumeIntegral<Dimension, HKernel<Dimension> >(*this,
                                                                 0.0,
                                                                 extent,
                                                                 10000);
    CHECK(mNs > 0.0);
  }
}

//------------------------------------------------------------------------------
// Return the gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
double
HKernel<Dimension>::gradValue(double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);

  const double lambda = this->kernelExtent();
  if (etaMagnitude < mInterval14) {
    return 0.5*mPeriod*cos(mPeriod*etaMagnitude - mOffset);
  } else if (etaMagnitude <= mInterval34) {
    return 0.0;
  } else if (etaMagnitude < lambda) {
    return 0.5*mPeriod*cos(mPeriod*etaMagnitude - mOffset);
  } else {
    return 0.0;
  }
}

//------------------------------------------------------------------------------
// Return the second derivative value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
double
HKernel<Dimension>::grad2Value(double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);

  const double lambda = this->kernelExtent();
  if (etaMagnitude < mInterval14) {
    return -0.5*mPeriod*mPeriod*sin(mPeriod*etaMagnitude - mOffset);
  } else if (etaMagnitude <= mInterval34) {
    return 0.0;
  } else if (etaMagnitude < lambda) {
    return -0.5*mPeriod*mPeriod*sin(mPeriod*etaMagnitude - mOffset);
  } else {
    return 0.0;
  }
}

//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
template class HKernel< Dim<1> >;
template class HKernel< Dim<2> >;
template class HKernel< Dim<3> >;

}

