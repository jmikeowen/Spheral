//---------------------------------Spheral++----------------------------------//
// bisectRoot -- implements the straight-up bisection root-finding search.
//           This is a stripped down version of our newtonRaphson routine.
//
// Created by JMO, Thu Oct 27 10:20:22 PDT 2016
//----------------------------------------------------------------------------//
#ifndef __Spheral_bisectRoot_hh__
#define __Spheral_bisectRoot_hh__

#include <map>
#include "Utilities/DBC.hh"
#include "Utilities/SpheralFunctions.hh"

namespace Spheral {

template<typename Function>
inline
double
bisectRoot(const Function& functor,
           double x1,
           double x2,
           const double xaccuracy = 1.0e-15,
           const double yaccuracy = 1.0e-15,
           const unsigned maxIterations = 100) {

  // Initialize values for the function and its derivative.
  double xminValue = functor(x1);
  double xmaxValue = functor(x2);

  // Is the root already at the min or max range?
  if (fuzzyEqual(xminValue, 0.0, yaccuracy)) return x1;
  if (fuzzyEqual(xmaxValue, 0.0, yaccuracy)) return x2;

  // Make sure the root is bracketed by the input range.
  VERIFY2(distinctlyLessThan(xminValue * xmaxValue, 0.0),
          "bisectRoot: root must be bracketed by input range:  " << xminValue << " " << xmaxValue);

  // Initialize the searching parameters.
  double xl, xh;
  if (xminValue < 0.0) {
    xl = x1;
    xh = x2;
  } else {
    CHECK(xminValue > 0.0 && xmaxValue < 0.0);
    xl = x2;
    xh = x1;
  }
  double rootSafe = 0.5*(x1 + x2);
  double dxold = std::abs(x2 - x1);
  double dx = dxold;
  double f = functor(rootSafe);

  // Iterate until we either converge or achieve the desired accuracy.
  unsigned iter = 0;
  while (iter < maxIterations) {
    ++iter;
    dxold = dx;
    dx = 0.5*(xh - xl);
    rootSafe = xl + dx;
    if (fuzzyEqual(xl, rootSafe, xaccuracy)) return rootSafe;
    if (std::abs(dx) <= xaccuracy) return rootSafe;

    f = functor(rootSafe);
    if (f < 0.0) {
      xl = rootSafe;
    } else {
      xh = rootSafe;
    }

  }

  VERIFY2(false, "bisectRoot: failed to converge!");

}

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Function>
  double
  bisectRoot(const Function& functor,
             double x1,
             double x2,
             const double xaccuracy = 1.0e-15,
             const double yaccuracy = 1.0e-15,
             const unsigned maxIterations = 100);
}

#endif
