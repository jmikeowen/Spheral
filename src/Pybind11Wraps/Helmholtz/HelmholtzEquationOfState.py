#-------------------------------------------------------------------------------
# HelmholtzEquationOfState
#-------------------------------------------------------------------------------
from PYB11Generator import *
from EquationOfState import *
from EOSAbstractMethods import *

@PYB11template("Dimension")
class HelmholtzEquationOfState(EquationOfState):

    typedefs = """
    typedef typename %(Dimension)s::Scalar Scalar;
    typedef Field<%(Dimension)s, Scalar> ScalarField;
"""

    #...........................................................................
    # Constructors
    def pyinit(self,
               constants = "const PhysicalConstants&",
               minimumPressure = ("const double", "-std::numeric_limits<double>::max()"),
               maximumPressure = ("const double",  "std::numeric_limits<double>::max()"),
               minimumTemperature = ("const double", "-std::numeric_limits<double>::min()"),
               minPressureType = ("const MaterialPressureMinType", "MaterialPressureMinType::PressureFloor"),
               abar0 = ("double", "13.6"),
               zbar0 = ("double", "6.8")):
        "Helmholtz constructor"

    #...........................................................................
    # Properties
    abar = PYB11property(returnpolicy="reference_internal")
    zbar = PYB11property(returnpolicy="reference_internal")
    needUpdate = PYB11property("bool", "getUpdateStatus", "setUpdateStatus")
    
#-------------------------------------------------------------------------------
# Add the virtual interface
#-------------------------------------------------------------------------------
PYB11inject(EOSAbstractMethods, HelmholtzEquationOfState, virtual=True)
