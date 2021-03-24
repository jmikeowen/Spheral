//---------------------------------Spheral++----------------------------------//
// QuadraticInterpolator
//
// Encapsulates the algorithm and data for parabolic interpolation in 1D
// Assumes the results is interpolated as y_interp = a + b*x + c*x^2
//
// Created by JMO, Fri Dec  4 14:28:08 PST 2020
//----------------------------------------------------------------------------//
#include "QuadraticInterpolator.hh"

#include <Eigen/Dense>

namespace Spheral {

//------------------------------------------------------------------------------
// Default constructor
//------------------------------------------------------------------------------
QuadraticInterpolator::QuadraticInterpolator():
  mN1(),
  mXmin(),
  mXmax(),
  mXstep(),
  mcoeffs() {
}

//------------------------------------------------------------------------------
// Construct with tabulated data
//------------------------------------------------------------------------------
QuadraticInterpolator::QuadraticInterpolator(const double xmin,
                                             const double xmax,
                                             const std::vector<double>& yvals):
  mN1(),
  mXmin(),
  mXmax(),
  mXstep(),
  mcoeffs() {
  this->initialize(xmin, xmax, yvals);
}

//------------------------------------------------------------------------------
// Initialize the interpolation to fit the given data
// Note, because we're doing a parabolic fit there are 2 fewer sets of coeficients
// than the size of the table we're fitting.
//------------------------------------------------------------------------------
void
QuadraticInterpolator::initialize(const double xmin,
                                  const double xmax,
                                  const std::vector<double>& yvals) {
  const auto n = yvals.size();
  REQUIRE(n > 2);      // Need at least 3 points to fit a parabola
  REQUIRE(xmax > xmin);

  mN1 = n - 3;  // Maximum index into arrays
  mXmin = xmin;
  mXmax = xmax;
  mXstep = (xmax - xmin)/(n - 1);
  mcoeffs.resize(3*(n - 2));

  typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> EMatrix;
  typedef Eigen::Matrix<double, 3, 1> EVector;

  // We use simple least squares fitting for each 3-point interval, giving us an
  // exact parabolic fit for those three points.
  // Find the coefficient fits.
  double x0, x1, x2;
  EMatrix A;
  EVector B, C;
  for (auto i0 = 0u; i0 < n - 2; ++i0) {
    const auto i1 = i0 + 1u;
    const auto i2 = i0 + 2u;
    CHECK(i2 < n);
    x0 = xmin + i0*mXstep;
    x1 = x0 + mXstep;
    x2 = x1 + mXstep;
    A << 1.0, x0, x0*x0,
         1.0, x1, x1*x1,
         1.0, x2, x2*x2;
    B << yvals[i0], yvals[i1], yvals[i2];
    C = A.inverse()*B;
    mcoeffs[3*i0    ] = C(0);
    mcoeffs[3*i0 + 1] = C(1);
    mcoeffs[3*i0 + 2] = C(2);
  }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
QuadraticInterpolator::~QuadraticInterpolator() {
}

}
