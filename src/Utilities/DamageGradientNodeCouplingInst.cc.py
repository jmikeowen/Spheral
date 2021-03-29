text = """
//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "Utilities/DamageGradientNodeCoupling.cc"

namespace Spheral {
  template class DamageGradientNodeCoupling< Dim< %(ndim)s > >;
}
"""
