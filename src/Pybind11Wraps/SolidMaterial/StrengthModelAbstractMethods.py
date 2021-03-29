#-------------------------------------------------------------------------------
# Abstract methods required of StrengthModels
#-------------------------------------------------------------------------------
from PYB11Generator import *

@PYB11ignore
class StrengthModelAbstractMethods:

    @PYB11const
    def shearModulus(self,
                     shearModulus = "Field<%(Dimension)s, Scalar>&",
                     density = "const Field<%(Dimension)s, Scalar>&",
                     specificThermalEnergy = "const Field<%(Dimension)s, Scalar>&",
                     pressure = "const Field<%(Dimension)s, Scalar>&",
                     damage = "const Field<%(Dimension)s, SymTensor>&"):
        return "void"

    @PYB11const
    def yieldStrength(self,
                      yieldStrength = "Field<%(Dimension)s, Scalar>&",
                      density = "const Field<%(Dimension)s, Scalar>&",
                      specificThermalEnergy = "const Field<%(Dimension)s, Scalar>&",
                      pressure = "const Field<%(Dimension)s, Scalar>&",
                      plasticStrain = "const Field<%(Dimension)s, Scalar>&",
                      plasticStrainRate = "const Field<%(Dimension)s, Scalar>&",
                      damage = "const Field<%(Dimension)s, SymTensor>&"):
        return "void"
