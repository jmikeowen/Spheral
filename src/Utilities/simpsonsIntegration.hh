//---------------------------------Spheral++----------------------------------//
// simpsonsIntegration
//
// Implements the Simpson's rule numerical integration method.
//
// Created by JMO, Fri May  6 21:24:44 PDT 2011
//----------------------------------------------------------------------------//
#ifndef __Spheral_simpsonsIntegration__
#define __Spheral_simpsonsIntegration__

#include "Utilities/DBC.hh"
#include "Utilities/DataTypeTraits.hh"

namespace Spheral {

template<typename Function, typename Result, typename Value>
inline
Result
simpsonsIntegration(const Function& function,
                    const Value x0,
                    const Value x1,
                    const unsigned numBins) {

  // Pre-conditions.
  VERIFY2(x0 <= x1, "Require integration range ordered:  " << x0 << " !< " << x1);
  VERIFY2(numBins > 1 and numBins % 2 == 0, "Require numBins a non-zero multiple of 2.");

  // Possible quick answer?
  if (x0 == x1) return 0.0;

  // Prepare our variables.
  unsigned i;
  Result integrand, result = DataTypeTraits<Value>::zero();

  // Size of the bins.
  const Value dx = (x1 - x0)/numBins;

  // Walk the bins and accumulate the answer.
  for (i = 0u; i < numBins + 1u; ++i) {
    integrand = function(x0 + i*dx);
    if (i == 0 or i == numBins) {
      result += integrand;
    } else if (i % 2 == 0) {
      result += 2.0*integrand;
    } else {
      result += 4.0*integrand;
    }
  }
    
  // Finish up and return the result.
  result *= (x1 - x0)/(3.0*numBins);
  return result;
}

}

#else

// Forward declaration.
namespace Spheral {
  template<typename Function, typename Result, typename Value>
  Result
  simpsonsIntegration(const Function& function,
                      const Value x0,
                      const Value x1,
                      const unsigned numBins);
}

#endif
