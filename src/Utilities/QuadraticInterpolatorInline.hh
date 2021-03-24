#include "Utilities/DBC.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Interpolate for the given x value.
//------------------------------------------------------------------------------
inline
double
QuadraticInterpolator::operator()(const double x) const {
  const auto i0 = lowerBound(x);
  return mcoeffs[i0] + (mcoeffs[i0 + 1] + mcoeffs[i0 + 2]*x)*x;
}

//------------------------------------------------------------------------------
// Interpolate the first derivative the given x value.
//------------------------------------------------------------------------------
inline
double
QuadraticInterpolator::prime(const double x) const {
  const auto i0 = lowerBound(x);
  return mcoeffs[i0 + 1] + 2.0*mcoeffs[i0 + 2]*x;
}

//------------------------------------------------------------------------------
// Interpolate the second derivative for the given x value.
// Just a constant value, so not a great fit.
//------------------------------------------------------------------------------
inline
double
QuadraticInterpolator::prime2(const double x) const {
  const auto i0 = lowerBound(x);
  return 2.0*mcoeffs[i0 + 2];
}

//------------------------------------------------------------------------------
// Return the lower bound entry in the table for the given x coordinate
//------------------------------------------------------------------------------
inline
size_t
QuadraticInterpolator::lowerBound(const double x) const {
  const auto result = 3u*std::min(mN1, size_t(std::max(0.0, x - mXmin)/mXstep));
  ENSURE(result <= 3u*mN1);
  return result;
}

//------------------------------------------------------------------------------
// Data accessors
//------------------------------------------------------------------------------
inline
size_t
QuadraticInterpolator::size() const {
  return mcoeffs.size();
}

inline
double
QuadraticInterpolator::xmin() const {
  return mXmin;
}

inline
double
QuadraticInterpolator::xmax() const {
  return mXmax;
}

inline
double
QuadraticInterpolator::xstep() const {
  return mXstep;
}

inline
const std::vector<double>&
QuadraticInterpolator::coeffs() const {
  return mcoeffs;
}

}
