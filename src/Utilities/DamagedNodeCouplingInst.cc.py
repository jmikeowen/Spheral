text = """
//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "Utilities/DamagedNodeCoupling.cc"

namespace Spheral {
  template class DamagedNodeCoupling< Dim< %(ndim)s > >;
}
"""
