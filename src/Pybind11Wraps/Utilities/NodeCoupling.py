#-------------------------------------------------------------------------------
# NodeCoupling
#-------------------------------------------------------------------------------
from PYB11Generator import *

@PYB11module("SpheralUtilities")
class NodeCoupling:
    "A functor base class encapsulating how we couple pairs of nodes."

    def pyinit(self):
        "Default constructor"

    @PYB11virtual
    @PYB11const
    @PYB11cppname("operator()")
    def __call__(self,
                 pair = "const NodePairIdxType&"):
        "Functional method to override for coupling (nodeListi, i) <-> (nodeListj, j)"
        return "double"

#-------------------------------------------------------------------------------
# DamagedNodeCoupling
#-------------------------------------------------------------------------------
@PYB11template("Dimension")
class DamagedNodeCoupling(NodeCoupling):
    """A functor class encapsulating how we couple solid nodes in the presence of
multiple materials and damage.

This form simply directly damages each pair based on their mutual damage."""

    PYB11typedefs = """
  typedef typename %(Dimension)s::Scalar Scalar;
  typedef typename %(Dimension)s::Vector Vector;
  typedef typename %(Dimension)s::Tensor Tensor;
  typedef typename %(Dimension)s::SymTensor SymTensor;
"""

    def pyinit(self,
               state = "const State<%(Dimension)s>&",
               pairs = "NodePairList&"):
        "Constructor"

#-------------------------------------------------------------------------------
# DamageGradientNodeCoupling
#-------------------------------------------------------------------------------
@PYB11template("Dimension")
class DamageGradientNodeCoupling(NodeCoupling):
    """A functor class encapsulating how we couple solid nodes in the presence of
multiple materials and damage.

This one attempts to mock up the shielding effect of ThreePointDamagedNodeCoupling
by using local damage gradient to estimate when nodes are separated by
regions of greater damage (or fractures)."""

#-------------------------------------------------------------------------------
# ThreePointDamagedNodeCoupling
#-------------------------------------------------------------------------------
@PYB11template("Dimension")
class ThreePointDamagedNodeCoupling(NodeCoupling):
    """A functor class encapsulating how we couple solid nodes in the presence of
multiple materials and damage.

This for uses the "three point" formalism, which allows damaged points to
cut communication between pairs that talk across them."""

    PYB11typedefs = """
  typedef typename %(Dimension)s::Scalar Scalar;
  typedef typename %(Dimension)s::Vector Vector;
  typedef typename %(Dimension)s::Tensor Tensor;
  typedef typename %(Dimension)s::SymTensor SymTensor;
"""

    def pyinit(self,
               state = "const State<%(Dimension)s>&",
               W = "const TableKernel<%(Dimension)s>&",
               pairs = "NodePairList&"):
        "Constructor"
