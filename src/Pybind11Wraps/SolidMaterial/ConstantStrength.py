#-------------------------------------------------------------------------------
# ConstantStrength
#-------------------------------------------------------------------------------
from PYB11Generator import *
from StrengthModel import *
from StrengthModelAbstractMethods import *

@PYB11template("Dimension")
@PYB11module("SpheralSolidMaterial")
class ConstantStrength(StrengthModel):
    """ConstantStrength -- An implentation of StrengthModel returning constant
values for the shear modulus and yield strength."""

    PYB11typedefs = """
    using Scalar = typename %(Dimension)s::Scalar;
    using SymTensor = typename %(Dimension)s::SymTensor;
    using ScalarField = Field<%(Dimension)s, Scalar>;
"""

    #...........................................................................
    # Constructors
    def pyinit(self,
               mu0 = "const double",
               Y0 = "const double"):
        "Construct with constant shear modulus (mu0) and yield strength (Y0)"

    def pyinit1(self,
                mu0 = "const double",
                Y0 = "const double",
                eos = "const SolidEquationOfState<%(Dimension)s>&"):
        "Construct with constant shear modulus (mu0), yield strength (Y0), and associated solid EOS"

    #...........................................................................
    # Properties
    mu0 = PYB11property("double", doc="shear modulus")
    Y0 = PYB11property("double", doc="yield strength")

#-------------------------------------------------------------------------------
# Inject Strength interface
#-------------------------------------------------------------------------------
PYB11inject(StrengthModelAbstractMethods, ConstantStrength, virtual=True, pure_virtual=False)
