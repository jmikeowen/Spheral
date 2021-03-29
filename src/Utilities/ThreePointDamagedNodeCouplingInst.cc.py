text = """
//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "Utilities/ThreePointDamagedNodeCoupling.cc"

namespace Spheral {
  template class ThreePointDamagedNodeCoupling< Dim< %(ndim)s > >;
}
"""
