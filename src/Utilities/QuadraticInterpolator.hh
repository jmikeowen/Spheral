//---------------------------------Spheral++----------------------------------//
// QuadraticInterpolator
//
// Encapsulates the algorithm and data for parabolic interpolation in 1D
// Assumes the results is interpolated as y_interp = a + b*x + c*x^2
//
// Created by JMO, Fri Dec  4 14:28:08 PST 2020
//----------------------------------------------------------------------------//
#ifndef __Spheral_QuadraticInterpolator__
#define __Spheral_QuadraticInterpolator__

#include <vector>

namespace Spheral {

class QuadraticInterpolator {
public:
  //--------------------------- Public Interface ---------------------------//
  // Constructors, destructors
  QuadraticInterpolator(const double xmin,
                        const double xmax,
                        const std::vector<double>& yvals);
  QuadraticInterpolator();
  ~QuadraticInterpolator();

  // Initialize for interpolating in the given data
  void initialize(const double xmin,
                  const double xmax,
                  const std::vector<double>& yvals);

  // Interpolate for the y value
  double operator()(const double x) const;
  double prime(const double x) const;    // First derivative
  double prime2(const double x) const;   // Second derivative

  // Return the lower bound index in the table for the given x coordinate
  size_t lowerBound(const double x) const;

  // Allow read access the internal data representation
  size_t size() const;                        // The size of the tabulated coefficient arrays
  double xmin() const;                        // Minimum x coordinate for table              
  double xmax() const;                        // Maximum x coordinate for table              
  double xstep() const;                       // delta x between tabulated values            
  const std::vector<double>& coeffs() const;  // the fitting coefficients
  
private:
  //--------------------------- Private Interface --------------------------//
  // Member data
  size_t mN1;
  double mXmin, mXmax, mXstep;
  std::vector<double> mcoeffs;
};

}

#include "QuadraticInterpolatorInline.hh"

#else

// Forward declaration
namespace Spheral {
  class QuadraticInterpolator;
}

#endif
