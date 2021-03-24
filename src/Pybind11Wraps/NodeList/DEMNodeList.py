from PYB11Generator import *
from NodeList import NodeList
from RestartMethods import *

#-------------------------------------------------------------------------------
# FluidNodeList template
#-------------------------------------------------------------------------------
@PYB11template("Dimension")
@PYB11module("SpheralNodeList")
@PYB11dynamic_attr
class DEMNodeList(NodeList):
    "Spheral DEMNodeList base class in %(Dimension)s, i.e.,  the NodeList for Discrete Element Modelling."

    PYB11typedefs = """
    typedef typename %(Dimension)s::Scalar Scalar;
    typedef typename %(Dimension)s::Vector Vector;
    typedef typename %(Dimension)s::Tensor Tensor;
    typedef typename %(Dimension)s::SymTensor SymTensor;
    typedef Field<%(Dimension)s, Scalar> ScalarField;
    typedef Field<%(Dimension)s, Vector> VectorField;
    typedef Field<%(Dimension)s, Tensor> TensorField;
    typedef Field<%(Dimension)s, SymTensor> SymTensorField;
"""

    def pyinit(self,
               name = "std::string",
               numInternal = ("int", "0"),
               numGhost = ("int", "0"),
               hmin = ("double", "1e-20"),
               hmax = ("double", "1e20"),
               hminratio = ("double", "0.1"),
               nPerh = ("double", "2.01"),
               maxNumNeighbors = ("int", "500")):
        "Constructor for a DEMNodeList class."
        return

    @PYB11const
    @PYB11returnpolicy("reference_internal")
    def angularVelocity(self):
        "The angular velocity field"
        return "const VectorField&"


    @PYB11pycppname("angularVelocity")
    def setangularVelocity(self, val="const VectorField&"):
        "Set the angular velocity"
        return "void"

    #@PYB11virtual
    #@PYB11const
    #def volume(self, result="ScalarField&"):
    #    "Compute the current volume, storing the result in the argument ScalarField."
    #    return "void"

    #@PYB11virtual
    #@PYB11const
    #def linearMomentum(self, result="VectorField&"):
    #    "Compute the current linear momentum, storing the result in the argument ScalarField."
    #    return "void"

    #@PYB11virtual
    #@PYB11const
    #def totalEnergy(self, result="ScalarField&"):
    #    "Compute the current total energy, storing the result in the argument ScalarField."
    #    return "void"


    #...........................................................................
    # Comparison
    def __eq__(self):
        "Equivalence test with another DEMNodeList"

    def __ne__(self):
        "Inequivalence test with another DEMNodeList"

    #...........................................................................
    # Properties
    #rhoMin = PYB11property("Scalar", "rhoMin", "rhoMin", doc="The minimum allowed mass density.")
    #rhoMax = PYB11property("Scalar", "rhoMax", "rhoMax", doc="The maximum allowed mass density.")

#-------------------------------------------------------------------------------
# Inject the restart methods
#-------------------------------------------------------------------------------
PYB11inject(RestartMethods, DEMNodeList)
