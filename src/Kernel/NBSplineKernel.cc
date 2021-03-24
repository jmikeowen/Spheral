//---------------------------------Spheral++----------------------------------//
// NBSplineKernel -- The nth order B Spline kernel.
//
// Schoenberg 1969, Journal of Approximation Theory, 2, 167-206.
//
// Created by JMO, Mon Jan 13 22:24:23 PST 2003
//----------------------------------------------------------------------------//
#include "NBSplineKernel.hh"
#include "Utilities/DBC.hh"

namespace Spheral {


//------------------------------------------------------------------------------
// Return the kernel weight for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
double
NBSplineKernel<Dimension>::kernelValue(double eta, const double Hdet) const {
  REQUIRE(eta >= 0.0);
  REQUIRE(Hdet >= 0.0);

  if (eta >= this->kernelExtent()) {
    return 0.0;

  } else {

    const int k = order() + 1;
    const int k1 = std::max(0, k - 1);
    const double halfk = 0.5*k;

    double result = 0.0;
    for (int i = 0; i <= k; ++i) {
      result += pow(-1.0, i)*binomialCoefficient(k, i)*
        oneSidedPowerFunction(eta - i + halfk, k1);
    }
    result *= this->volumeNormalization()*Hdet/factorial(k1);
    return result;
  }
}

//------------------------------------------------------------------------------
// Return the gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
double
NBSplineKernel<Dimension>::gradValue(double eta, const double Hdet) const {
  REQUIRE(eta >= 0.0);
  REQUIRE(Hdet >= 0.0);

  if (eta >= this->kernelExtent()) {
    return 0.0;

  } else {

    const int k = order() + 1;
    const int k2 = std::max(-1, k - 2);
    const double halfk = 0.5*k;

    double result = 0.0;
    for (int i = 0; i <= k; ++i) {
      result += pow(-1.0, i)*binomialCoefficient(k, i)*
        oneSidedPowerFunction(eta - i + halfk, k2);
    }
    result *= this->volumeNormalization()*Hdet/factorial(k2);
    return result;
  }
}

//------------------------------------------------------------------------------
// Return the second derivative value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
double
NBSplineKernel<Dimension>::grad2Value(double eta, const double Hdet) const {
  REQUIRE(eta >= 0.0);
  REQUIRE(Hdet >= 0.0);

  if (eta >= this->kernelExtent()) {
    return 0.0;

  } else {

    const int k = order() + 1;
    const int k3 = std::max(-2, k - 3);
    const double halfk = 0.5*k;

    double result = 0.0;
    for (int i = 0; i <= k; ++i) {
      result += pow(-1.0, i)*binomialCoefficient(k, i)*
        oneSidedPowerFunction(eta - i + halfk, k3);
    }
    result *= this->volumeNormalization()*Hdet/factorial(k3);
    return result;
  }
}

//------------------------------------------------------------------------------
// Set the kernels extent and volume normalization based upon the order.
//------------------------------------------------------------------------------
template<typename Dimension>
void
NBSplineKernel<Dimension>::
initializeKernel() {
  const int order = mOrder;
  REQUIRE(order >= 0);

  this->setKernelExtent((order + 1)/2);
  this->setInflectionPoint(0.0);

  // Numerically evaluate the volume normalization.
  this->setVolumeNormalization(1.0);
  double volumeIntegral = simpsonsVolumeIntegral<Dimension, NBSplineKernel<Dimension> >(*this, 0.0, this->kernelExtent(), 10000);
  CHECK(volumeIntegral > 0.0);
  this->setVolumeNormalization(1.0/volumeIntegral);
}

}
