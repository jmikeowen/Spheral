#-------------------------------------------------------------------------------
# PorousStrengthModel
#-------------------------------------------------------------------------------
from PYB11Generator import *
from StrengthModel import *
from StrengthModelAbstractMethods import *

@PYB11template("Dimension")
@PYB11module("SpheralSolidMaterial")
class PorousStrengthModel(StrengthModel):
    """PorousStrengthModel

An implementation of strain-alpha porosity model described in
Wunnemann, Collins, & Melosh, Icarus, 180, 514-527 (2006)
"A strain-based porosity model for use in hydrocode simulations of impacts
 and implications for transient crater growth in porous targets"

This model assumes you will provide a solid EOS which will be modified.
The underlying actualy solid EOS should provide the reference density, which
will be treated here as the compacted true solid reference density.

Note this model introduces a new state variable, the distention (alpha), which
the pressure now depends on.  This implies our usual definition of P(rho, eps)
now becomes P(rho, eps, alpha).  Our EOS interface does not recognize this
this parameter, so we store alpha locally and only allow Field updates of the
pressure (forbidding the single value P lookup the EOS usually allows)."""

    PYB11typedefs = """
    using Scalar = typename %(Dimension)s::Scalar;
    using SymTensor = typename %(Dimension)s::SymTensor;
    using ScalarField = Field<%(Dimension)s, Scalar>;
"""

    #...........................................................................
    # Constructors
    def pyinit(self,
               solidStrength = "const StrengthModel<%(Dimension)s>&"):
        "Construct with the strength model we're modifying"

    #...........................................................................
    # Virtual methods
    @PYB11virtual
    @PYB11const
    def providesSoundSpeed(self):
        return "bool"

    @PYB11virtual
    @PYB11const
    def providesBulkModulus(self):
        return "bool"

    @PYB11virtual
    @PYB11const
    def soundSpeed(self,
                   soundSpeed = "Field<%(Dimension)s, Scalar>&",
                   density = "const Field<%(Dimension)s, Scalar>&",
                   specificThermalEnergy = "const Field<%(Dimension)s, Scalar>&",
                   pressure = "const Field<%(Dimension)s, Scalar>&",
                   fluidSoundSpeed = "const Field<%(Dimension)s, Scalar>&",
                   damage = "const Field<%(Dimension)s, SymTensor>&"):
        return "void"

    @PYB11virtual
    @PYB11const
    def bulkModulus(self,
                    bulkModulus = "Field<%(Dimension)s, Scalar>&",
                    massDensity = "const Field<%(Dimension)s, Scalar>&",
                    specificThermalEnergy = "const Field<%(Dimension)s, Scalar>&"):
        return "void"

    @PYB11virtual
    @PYB11const
    def meltSpecificEnergy(self,
                           meltSpecificEnergy = "Field<%(Dimension)s, Scalar>&",
                           density = "const Field<%(Dimension)s, Scalar>&",
                           specficThermalEnergy = "const Field<%(Dimension)s, Scalar>&"):
        return "void"

    @PYB11virtual
    @PYB11const
    def coldSpecificEnergy(self,
                           coldSpecificEnergy = "Field<%(Dimension)s, Scalar>&",
                           density = "const Field<%(Dimension)s, Scalar>&",
                           specficThermalEnergy = "const Field<%(Dimension)s, Scalar>&"):
        return "void"

    #...........................................................................
    # Properties
    solidStrength = PYB11property("const StrengthModel<%(Dimension)s>&", returnpolicy="reference_internal")
    alpha = PYB11property("const Field<%(Dimension)s, Scalar>&", "alpha", "alpha", returnpolicy="reference_internal")

#-------------------------------------------------------------------------------
# Inject abstract interface
#-------------------------------------------------------------------------------
PYB11inject(StrengthModelAbstractMethods, PorousStrengthModel, virtual=True, pure_virtual=False)
