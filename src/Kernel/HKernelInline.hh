#include "Geometry/Dimension.hh"
#include "TableKernel.hh"
#include "Utilities/DBC.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Return the kernel weight for a given normalized distance.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
HKernel<Dimension>::
kernelValue(double etaMagnitude, const double Hdet) const {
  REQUIRE(etaMagnitude >= 0.0);
  REQUIRE(Hdet >= 0.0);

  if (etaMagnitude <= mInterval34) {
    return 1.0;
  } else if (etaMagnitude < this->kernelExtent()) {
    return 0.5*(1.0 + sin(mPeriod*etaMagnitude - mOffset));
  } else {
    return 0.0;
  }

//   if (etaMagnitude < mInterval14) {
//     return 0.5*(1.0 + sin(mPeriod*etaMagnitude - mOffset));
//   } else if (etaMagnitude <= mInterval34) {
//     return 1.0;
//   } else if (etaMagnitude < kernelExtent()) {
//     return 0.5*(1.0 + sin(mPeriod*etaMagnitude - mOffset));
//   } else {
//     return 0.0;
//   }
}

//------------------------------------------------------------------------------
// Return the volume integral constant for use in the ideal H calculation.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
double
HKernel<Dimension>::
Ns() const {
  return mNs;
}

}
