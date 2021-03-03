from PYB11Generator import *
from FluidNodeList import FluidNodeList
from RestartMethods import *

#-------------------------------------------------------------------------------
# SolidNodeList template
#-------------------------------------------------------------------------------
@PYB11template("Dimension")
@PYB11module("SpheralNodeList")
@PYB11dynamic_attr
class SolidNodeList(FluidNodeList):
    "Spheral SolidNodeList base class in %(Dimension)s, i.e.,  the NodeList for solid dynamics."

    PYB11typedefs = """
    typedef typename %(Dimension)s::Scalar Scalar;
    typedef typename %(Dimension)s::Vector Vector;
    typedef typename %(Dimension)s::Tensor Tensor;
    typedef typename %(Dimension)s::SymTensor SymTensor;
    typedef Field<%(Dimension)s, int> IntField;
    typedef Field<%(Dimension)s, Scalar> ScalarField;
    typedef Field<%(Dimension)s, Vector> VectorField;
    typedef Field<%(Dimension)s, Tensor> TensorField;
    typedef Field<%(Dimension)s, SymTensor> SymTensorField;
"""

    def pyinit(self,
               name = "std::string",
               eos = "EquationOfState<%(Dimension)s>&",
               strength = "StrengthModel<%(Dimension)s>&",
               numInternal = ("int", "0"),
               numGhost = ("int", "0"),
               hmin = ("double", "1e-20"),
               hmax = ("double", "1e20"),
               hminratio = ("double", "0.1"),
               nPerh = ("double", "2.01"),
               maxNumNeighbors = ("int", "500"),
               rhoMin = ("double", "1e-10"),
               rhoMax = ("double", "1e100")):
        "Constructor for a SolidNodeList class."
        return

    @PYB11virtual
    @PYB11const
    def soundSpeed(self, result="ScalarField&"):
        "Compute the current sound speed, storing the result in the argument ScalarField."
        return "void"

    @PYB11virtual
    @PYB11const
    def bulkModulus(self, result="ScalarField&"):
        "Compute the current bulk modulus, storing the result in the argument ScalarField."
        return "void"

    @PYB11virtual
    @PYB11const
    def yieldStrength(self, result="ScalarField&"):
        "Compute the current yield strength, storing the result in the argument ScalarField."
        return "void"

    @PYB11const
    @PYB11returnpolicy("reference_internal")
    def deviatoricStress(self):
        "The mass density field"
        return "const SymTensorField&"

    @PYB11const
    @PYB11returnpolicy("reference_internal")
    def plasticStrain(self):
        "The plastic strain field"
        return "const ScalarField&"

    @PYB11const
    @PYB11returnpolicy("reference_internal")
    def plasticStrainRate(self):
        "The plastic strain rate field"
        return "const ScalarField&"

    @PYB11const
    @PYB11returnpolicy("reference_internal")
    def damage(self):
        "The damage field"
        return "const SymTensorField&"

    @PYB11const
    @PYB11returnpolicy("reference_internal")
    def fragmentIDs(self):
        "The fragment IDs field"
        return "const IntField&"

    @PYB11const
    @PYB11returnpolicy("reference_internal")
    def particleTypes(self):
        "The particle type field"
        return "const IntField&"

    @PYB11const
    @PYB11returnpolicy("reference_internal")
    def strengthModel(self):
        "Return the strength model object this SolidNodeList is associated with."
        return "const StrengthModel<%(Dimension)s>&"

    # Comparison
    def __eq__(self):
        "Equivalence test with another SolidNodeList"

    def __ne__(self):
        "Inequivalence test with another SolidNodeList"

#-------------------------------------------------------------------------------
# Inject the restart methods
#-------------------------------------------------------------------------------
PYB11inject(RestartMethods, SolidNodeList)
