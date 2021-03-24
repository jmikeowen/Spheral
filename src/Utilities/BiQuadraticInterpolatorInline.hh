#include "Utilities/DBC.hh"
#include <Eigen/Dense>
#include <iostream>

namespace Spheral {

//------------------------------------------------------------------------------
// Default constructor
//------------------------------------------------------------------------------
BiQuadraticInterpolator::BiQuadraticInterpolator():
  mnx1(),
  mny1(),
  mxmin(),
  mxmax(),
  mxstep(),
  mcoeffs() {
}

//------------------------------------------------------------------------------
// Construct with tabulated data
//------------------------------------------------------------------------------
template<typename Func>
BiQuadraticInterpolator::BiQuadraticInterpolator(const Vector& xmin,
                                                 const Vector& xmax,
                                                 const size_t nx,
                                                 const size_t ny,
                                                 const Func& F):
  mnx1(),
  mny1(),
  mxmin(),
  mxmax(),
  mxstep(),
  mcoeffs() {
  this->initialize(xmin, xmax, nx, ny, F);
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
BiQuadraticInterpolator::~BiQuadraticInterpolator() {
}

//------------------------------------------------------------------------------
// Initialize the interpolation to fit the given data
//------------------------------------------------------------------------------
template<typename Func>
void
BiQuadraticInterpolator::initialize(const Vector& xmin,
                                    const Vector& xmax,
                                    const size_t nx,
                                    const size_t ny,
                                    const Func& F) {

  // Size stuff up.
  REQUIRE(nx > 2u);
  REQUIRE(ny > 2u);
  mnx1 = nx - 2u;
  mny1 = ny - 2u;
  mcoeffs.resize(6*mnx1*mny1);

  // Figure out the sampling steps.
  mxmin = xmin;
  mxmax = xmax;
  mxstep = {(xmax[0] - xmin[0])/mnx1,
            (xmax[1] - xmin[1])/mny1};

  // Fit the coefficients
  Vector x00, x01, x02, x10, x11, x12, x20, x21, x22;
  Eigen::MatrixXd A(9, 6);
  Eigen::VectorXd b(9), c(9);
  for (auto i = 0u; i < mnx1; ++i) {
    for (auto j = 0u; j < mny1; ++j) {
      x00 = {xmin[0] + i      *mxstep[0], xmin[1] + j      *mxstep[1]};
      x10 = {xmin[0] + (i + 1)*mxstep[0], xmin[1] + j      *mxstep[1]};
      x20 = {xmin[0] + (i + 2)*mxstep[0], xmin[1] + j      *mxstep[1]};
      x01 = {xmin[0] + i      *mxstep[0], xmin[1] + (j + 1)*mxstep[1]};
      x11 = {xmin[0] + (i + 1)*mxstep[0], xmin[1] + (j + 1)*mxstep[1]};
      x21 = {xmin[0] + (i + 2)*mxstep[0], xmin[1] + (j + 1)*mxstep[1]};
      x02 = {xmin[0] + i      *mxstep[0], xmin[1] + (j + 2)*mxstep[1]};
      x12 = {xmin[0] + (i + 1)*mxstep[0], xmin[1] + (j + 2)*mxstep[1]};
      x22 = {xmin[0] + (i + 2)*mxstep[0], xmin[1] + (j + 2)*mxstep[1]};
      A << 1.0, x00[0], x00[1], x00[0]*x00[1], x00[0]*x00[0], x00[1]*x00[1],
           1.0, x01[0], x01[1], x01[0]*x01[1], x01[0]*x01[0], x01[1]*x01[1],
           1.0, x02[0], x02[1], x02[0]*x02[1], x02[0]*x02[0], x02[1]*x02[1],
           1.0, x10[0], x10[1], x10[0]*x10[1], x10[0]*x10[0], x10[1]*x10[1],
           1.0, x11[0], x11[1], x11[0]*x11[1], x11[0]*x11[0], x11[1]*x11[1],
           1.0, x12[0], x12[1], x12[0]*x12[1], x12[0]*x12[0], x12[1]*x12[1],
           1.0, x20[0], x20[1], x20[0]*x20[1], x20[0]*x20[0], x20[1]*x20[1],
           1.0, x21[0], x21[1], x21[0]*x21[1], x21[0]*x21[0], x21[1]*x21[1],
           1.0, x22[0], x22[1], x22[0]*x22[1], x22[0]*x22[0], x22[1]*x22[1];
      b << F(x00), F(x01), F(x02), F(x10), F(x11), F(x12), F(x20), F(x21), F(x22);
      c = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
      // std::cerr << "------------------------------------------------------------------------------\n"
      //           << "x00: " << x00 << "\n"
      //           << "x10: " << x10 << "\n"
      //           << "x20: " << x20 << "\n"
      //           << "x01: " << x01 << "\n"
      //           << "x11: " << x11 << "\n"
      //           << "x21: " << x21 << "\n"
      //           << "x02: " << x02 << "\n"
      //           << "x12: " << x12 << "\n"
      //           << "x22: " << x22 << "\n"
      //           << "A:\n" << A << "\n"
      //           << "b:\n" << b << "\n"
      //           << "c:\n" << c << "\n";
      const auto k = 6*(i + j*mnx1);
      mcoeffs[k    ] = c(0);
      mcoeffs[k + 1] = c(1);
      mcoeffs[k + 2] = c(2);
      mcoeffs[k + 3] = c(3);
      mcoeffs[k + 4] = c(4);
      mcoeffs[k + 5] = c(5);
    }
  }
}

//------------------------------------------------------------------------------
// Interpolate for the given coordinate.
//------------------------------------------------------------------------------
inline
double
BiQuadraticInterpolator::operator()(const Vector& pos) const {
  const auto x = std::max(mxmin[0], std::min(mxmax[0], pos[0]));
  const auto y = std::max(mxmin[1], std::min(mxmax[1], pos[1]));
  const auto i0 = lowerBound(x, y);
  // std::cerr << "================================================================================\n"
  //           << "mxlog, mylog : " << mxlog << " " << mylog << "\n"
  //           << "pos   : " << pos << "\n"
  //           << "(x,y) : " << x << " " << y << "\n"
  //           << "i0    : " << i0 << "\n"
  //           << "coeffs: " << mcoeffs[i0] << " " << mcoeffs[i0 + 1] << " " << mcoeffs[i0 + 2] << " " << mcoeffs[i0 + 3] << " " << mcoeffs[i0 + 4] << " " << mcoeffs[i0 + 5] << "\n"
  //           << "F(x,y): " << mcoeffs[i0] + mcoeffs[i0 + 1]*x + mcoeffs[i0 + 2]*y + mcoeffs[i0 + 3]*x*y + mcoeffs[i0 + 4]*x*x + mcoeffs[i0 + 5]*y*y << "\n";
  return mcoeffs[i0] + mcoeffs[i0 + 1]*x + mcoeffs[i0 + 2]*y + mcoeffs[i0 + 3]*x*y + mcoeffs[i0 + 4]*x*x + mcoeffs[i0 + 5]*y*y;
}

//------------------------------------------------------------------------------
// Interpolate for the gradient (x)
//------------------------------------------------------------------------------
inline
double
BiQuadraticInterpolator::prime_x(const Vector& pos) const {
  const auto x = std::max(mxmin[0], std::min(mxmax[0], pos[0]));
  const auto y = std::max(mxmin[1], std::min(mxmax[1], pos[1]));
  const auto i0 = lowerBound(x, y);
  return mcoeffs[i0 + 1] + mcoeffs[i0 + 3]*y + 2.0*mcoeffs[i0 + 4]*x;
}

//------------------------------------------------------------------------------
// Interpolate for the gradient (y)
//------------------------------------------------------------------------------
inline
double
BiQuadraticInterpolator::prime_y(const Vector& pos) const {
  const auto x = std::max(mxmin[0], std::min(mxmax[0], pos[0]));
  const auto y = std::max(mxmin[1], std::min(mxmax[1], pos[1]));
  const auto i0 = lowerBound(x, y);
  return mcoeffs[i0 + 2] + mcoeffs[i0 + 3]*x + 2.0*mcoeffs[i0 + 5]*y;
}

//------------------------------------------------------------------------------
// Interpolate for the gradient2 (xx)
//------------------------------------------------------------------------------
inline
double
BiQuadraticInterpolator::prime2_xx(const Vector& pos) const {
  const auto x = std::max(mxmin[0], std::min(mxmax[0], pos[0]));
  const auto y = std::max(mxmin[1], std::min(mxmax[1], pos[1]));
  const auto i0 = lowerBound(x, y);
  return 2.0*mcoeffs[i0 + 4];
}

//------------------------------------------------------------------------------
// Interpolate for the gradient2 (xy)
//------------------------------------------------------------------------------
inline
double
BiQuadraticInterpolator::prime2_xy(const Vector& pos) const {
  const auto x = std::max(mxmin[0], std::min(mxmax[0], pos[0]));
  const auto y = std::max(mxmin[1], std::min(mxmax[1], pos[1]));
  const auto i0 = lowerBound(x, y);
  return mcoeffs[i0 + 3];
}

//------------------------------------------------------------------------------
// Interpolate for the gradient2 (yx)
//------------------------------------------------------------------------------
inline
double
BiQuadraticInterpolator::prime2_yx(const Vector& pos) const {
  const auto x = std::max(mxmin[0], std::min(mxmax[0], pos[0]));
  const auto y = std::max(mxmin[1], std::min(mxmax[1], pos[1]));
  const auto i0 = lowerBound(x, y);
  return mcoeffs[i0 + 3];
}

//------------------------------------------------------------------------------
// Interpolate for the gradient2 (yy)
//------------------------------------------------------------------------------
inline
double
BiQuadraticInterpolator::prime2_yy(const Vector& pos) const {
  const auto x = std::max(mxmin[0], std::min(mxmax[0], pos[0]));
  const auto y = std::max(mxmin[1], std::min(mxmax[1], pos[1]));
  const auto i0 = lowerBound(x, y);
  return 2.0*mcoeffs[i0 + 5];
}

//------------------------------------------------------------------------------
// Return the lower bound entry in the table for the given x coordinate
//------------------------------------------------------------------------------
inline
size_t
BiQuadraticInterpolator::lowerBound(const double x, const double y) const {
  const auto result = 6u*(mnx1*std::min(mny1 - 1u, size_t(std::max(0.0, y - mxmin[1])/mxstep[1])) +
                               std::min(mnx1 - 1u, size_t(std::max(0.0, x - mxmin[0])/mxstep[0])));
  ENSURE(result <= 6u*mnx1*mny1);
  return result;
}

//------------------------------------------------------------------------------
// Data accessors
//------------------------------------------------------------------------------
inline
size_t
BiQuadraticInterpolator::size() const {
  return mcoeffs.size();
}

inline
typename BiQuadraticInterpolator::Vector
BiQuadraticInterpolator::xmin() const {
  return mxmin;
}

inline
typename BiQuadraticInterpolator::Vector
BiQuadraticInterpolator::xmax() const {
  return mxmax;
}

inline
typename BiQuadraticInterpolator::Vector
BiQuadraticInterpolator::xstep() const {
  return mxstep;
}

inline
const std::vector<double>&
BiQuadraticInterpolator::coeffs() const {
  return mcoeffs;
}

}
