//---------------------------------Spheral++----------------------------------//
// BiQuadraticInterpolator
//
// Encapsulates the algorithm and data for parabolic interpolation in 2D
// Assumes the results is interpolated as
//   <F(x,y)> = c0 + c1*x + c2*y + c3*x*y + c4*x^2 + c5*y^2
//
// Created by JMO, Thu Dec 10 14:48:01 PST 2020
//----------------------------------------------------------------------------//
#ifndef __Spheral_BiQuadraticInterpolator__
#define __Spheral_BiQuadraticInterpolator__

#include "Geometry/Dimension.hh"
#include <vector>

namespace Spheral {

class BiQuadraticInterpolator {
public:
  //--------------------------- Public Interface ---------------------------//
  typedef Dim<2>::Vector Vector;

  // Constructors, destructors
  template<typename Func>
  BiQuadraticInterpolator(const Vector& xmin,
                          const Vector& xmax,
                          const size_t nx,
                          const size_t ny,
                          const Func& F);
  BiQuadraticInterpolator();
  ~BiQuadraticInterpolator();

  // Initialize for interpolating the given function
  template<typename Func>
  void initialize(const Vector& xmin,
                  const Vector& xmax,
                  const size_t nx,
                  const size_t ny,
                  const Func& F);

  // Interpolate for the F(x,y) value
  double operator()(const Vector& pos) const;

  // Interpolated gradient values.
  double prime_x(const Vector& pos) const;
  double prime_y(const Vector& pos) const;
  double prime2_xx(const Vector& pos) const;
  double prime2_xy(const Vector& pos) const;
  double prime2_yx(const Vector& pos) const;
  double prime2_yy(const Vector& pos) const;

  // Allow read access the internal data representation
  size_t size() const;                        // The size of the tabulated coefficient arrays
  Vector xmin() const;                        // Minimum coordinate for table              
  Vector xmax() const;                        // Maximum coordinate for table              
  Vector xstep() const;                       // Step size
  const std::vector<double>& coeffs() const;  // the fitting coefficients
  
private:
  //--------------------------- Private Interface --------------------------//
  // Member data
  size_t mnx1, mny1;
  Vector mxmin, mxmax, mxstep;
  std::vector<double> mcoeffs;

  // Return the lower bound index in the table of coefficients for the given position
  size_t lowerBound(const double x, const double y) const;
};

}

#include "BiQuadraticInterpolatorInline.hh"

#else

// Forward declaration
namespace Spheral {
  class BiQuadraticInterpolator;
}

#endif
