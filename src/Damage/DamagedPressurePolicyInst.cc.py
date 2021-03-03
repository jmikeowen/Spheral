text = """
//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "Geometry/Dimension.hh"
#include "Damage/DamagedPressurePolicy.cc"

namespace Spheral {
  template class DamagedPressurePolicy<Dim< %(ndim)s > >;
}
"""
