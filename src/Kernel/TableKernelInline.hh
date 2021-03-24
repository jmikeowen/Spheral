#include "Geometry/Dimension.hh"
#include "Utilities/DBC.hh"
#include "VolumeIntegrationFunctions.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Return the kernel weight for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
TableKernel<Dimension>::kernelValue(const double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);
  if (etaMagnitude < this->mKernelExtent) {
    return Hdet*mInterp(etaMagnitude);
  } else {
    return 0.0;
  }
}

//------------------------------------------------------------------------------
// Return the gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
TableKernel<Dimension>::gradValue(const double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);
  if (etaMagnitude < this->mKernelExtent) {
    return Hdet*mGradInterp(etaMagnitude);
  } else {
    return 0.0;
  }
}

//------------------------------------------------------------------------------
// Return the second derivative value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
TableKernel<Dimension>::grad2Value(const double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);
  if (etaMagnitude < this->mKernelExtent) {
    return Hdet*mGrad2Interp(etaMagnitude);
  } else {
    return 0.0;
  }
}

//------------------------------------------------------------------------------
// Return the kernel and gradient value for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
std::pair<double, double>
TableKernel<Dimension>::kernelAndGradValue(const double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);
  if (etaMagnitude < this->mKernelExtent) {
    return std::make_pair(Hdet*mInterp(etaMagnitude),
                          Hdet*mGradInterp(etaMagnitude));
  } else {
    return std::make_pair(0.0, 0.0);
  }
}

//------------------------------------------------------------------------------
// Return the kernel and gradient values for a set of normalized distances.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
void
TableKernel<Dimension>::kernelAndGradValues(const std::vector<double>& etaMagnitudes,
                                            const std::vector<double>& Hdets,
                                            std::vector<double>& kernelValues,
                                            std::vector<double>& gradValues) const {
  // Preconditions.
  const auto n = etaMagnitudes.size();
  BEGIN_CONTRACT_SCOPE
  {
    REQUIRE(Hdets.size() == n);
    for (auto i = 0u; i < n; ++i) {
      REQUIRE(etaMagnitudes[i] >= 0.0);
      REQUIRE(Hdets[i] >= 0.0);
    }
  }
  END_CONTRACT_SCOPE

  // Prepare the results.
  kernelValues.resize(n);
  gradValues.resize(n);

  // Fill those suckers in.
  for (auto i = 0u; i < n; ++i) {
    kernelValues[i] = Hdets[i]*mInterp(etaMagnitudes[i]);
    gradValues[i] = Hdets[i]*mGradInterp(etaMagnitudes[i]);
  }
}

//------------------------------------------------------------------------------
// Return the f1 RZ correctiong for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
TableKernel<Dimension>::f1(const double /*etaMagnitude*/) const {
  VERIFY2(false, "TableKernel::f1 lookup only valid for 2D kernels.");
}

template<>
inline
double
TableKernel<Dim<2> >::f1(const double etaMagnitude) const {
  REQUIRE(etaMagnitude >= 0.0);
  if (etaMagnitude < this->mKernelExtent) {
    return mf1Interp(etaMagnitude);
  } else {
    return 1.0;
  }
}

//------------------------------------------------------------------------------
// Return the f2 RZ correctiong for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
TableKernel<Dimension>::f2(const double /*etaMagnitude*/) const {
  VERIFY2(false, "TableKernel::f2 lookup only valid for 2D kernels.");
}

template<>
inline
double
TableKernel<Dim<2> >::f2(const double etaMagnitude) const {
  REQUIRE(etaMagnitude >= 0.0);
  return mf2Interp(etaMagnitude);
}

//------------------------------------------------------------------------------
// Return the grad_r f1 RZ correctiong for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
TableKernel<Dimension>::gradf1(const double /*etaMagnitude*/) const {
  VERIFY2(false, "TableKernel::gradf1 lookup only valid for 2D kernels.");
}

template<>
inline
double
TableKernel<Dim<2> >::gradf1(const double etaMagnitude) const {
  REQUIRE(etaMagnitude >= 0.0);
  return mf1Interp.prime(etaMagnitude);
}

//------------------------------------------------------------------------------
// Return the grad_r f2 RZ correctiong for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
TableKernel<Dimension>::gradf2(const double /*etaMagnitude*/) const {
  VERIFY2(false, "TableKernel::gradf2 lookup only valid for 2D kernels.");
}

template<>
inline
double
TableKernel<Dim<2> >::gradf2(const double etaMagnitude) const {
  REQUIRE(etaMagnitude >= 0.0);
  return mf2Interp.prime(etaMagnitude);
}

//------------------------------------------------------------------------------
// Return the (f1, f2, gradf1, gradf2) RZ values for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
void
TableKernel<Dimension>::f1Andf2(const double /*etaMagnitude*/,
                                double& /*f1*/,
                                double& /*f2*/,
                                double& /*gradf1*/,
                                double& /*gradf2*/) const {
  VERIFY2(false, "TableKernel::f1Andf2 lookup only valid for 2D kernels.");
}

template<>
inline
void
TableKernel<Dim<2> >::f1Andf2(const double etaMagnitude,
                              double& f1,
                              double& f2,
                              double& gradf1,
                              double& gradf2) const {
  REQUIRE(etaMagnitude >= 0.0);
  if (etaMagnitude < this->mKernelExtent) {
    f1 = mf1Interp(etaMagnitude);
    f2 = mf2Interp(etaMagnitude);
    gradf1 = mf1Interp.prime(etaMagnitude);
    gradf2 = mf2Interp.prime(etaMagnitude);
  } else {
    f1 = 1.0;
    f2 = 1.0;
    gradf1 = 0.0;
    gradf2 = 0.0;
  }
}

//------------------------------------------------------------------------------
// Return the assorted tabular lookup data.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
const std::vector<double>&
TableKernel<Dimension>::
nperhValues() const {
  return mNperhValues;
}

template<typename Dimension>
inline
const std::vector<double>&
TableKernel<Dimension>::
WsumValues() const {
  return mWsumValues;
}

template<typename Dimension>
inline
size_t
TableKernel<Dimension>::
numPoints() const {
  return mNumPoints;
}

}

