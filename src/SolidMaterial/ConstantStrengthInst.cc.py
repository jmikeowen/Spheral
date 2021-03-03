text = """
//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "SolidMaterial/ConstantStrength.cc"
#include "Geometry/Dimension.hh"

namespace Spheral {
  template class ConstantStrength<Dim< %(ndim)s > >;
}
"""
