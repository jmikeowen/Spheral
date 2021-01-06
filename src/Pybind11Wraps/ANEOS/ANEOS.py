#-------------------------------------------------------------------------------
# ANEOS
#-------------------------------------------------------------------------------
from PYB11Generator import *
from SolidEquationOfState import *
from EOSAbstractMethods import *

@PYB11template("Dimension")
class ANEOS(SolidEquationOfState):

    PYB11typedefs = """
    typedef typename %(Dimension)s::Scalar Scalar;
    typedef Field<%(Dimension)s, Scalar> ScalarField;
"""

    #...........................................................................
    # Constructors
    def pyinit(self,
               materialNumber = "int",
               numRhoVals = "unsigned",
               numTvals = "unsigned",
               rhoMin = "double", 
               rhoMax = "double",
               Tmin = "double", 
               Tmax = "double", 
               constants = "const PhysicalConstants&",
               externalPressure = ("const double", "0.0"),
               minimumPressure = ("const double", "-std::numeric_limits<double>::max()"),
               maximumPressure = ("const double",  "std::numeric_limits<double>::max()"),
               minPressureType = ("const MaterialPressureMinType", "MaterialPressureMinType::PressureFloor")):
        """ANEOS constructor

Note the material number is the "EOS#" from the associated ANEOS input file passed to initializeANEOS, and the 
negative of this number as passed in the array to initializeANEOS.

Also, note that (numRhoVals, numTvals) are used to compute an inverse lookup table in (rho, T)
for the specific thermal energy.  This table is currently linear in (rho, T), so be sure you both span
an appropriate range for (rhoMin, rhoMax), (Tmin, Tmax), *and* have enough values sampling in 
(numRhoVals, numTvals) to represent this inverse for your equation of state."""

    #...........................................................................
    # Methods
    @PYB11const
    def pressure(self,
                 massDensity = "const Scalar",
                 specificThermalEnergy = "const Scalar"):
        return "Scalar"

    @PYB11const
    def temperature(self,
                    massDensity = "const Scalar",
                    specificThermalEnergy = "const Scalar"):
        return "Scalar"

    @PYB11const
    def specificThermalEnergy(self,
                              massDensity = "const Scalar",
                              temperature = "const Scalar"):
        return "Scalar"

    @PYB11const
    def specificHeat(self,
                     massDensity = "const Scalar",
                     temperature = "const Scalar"):
        return "Scalar"

    @PYB11const
    def soundSpeed(self,
                   massDensity = "const Scalar",
                   specificThermalEnergy = "const Scalar"):
        return "Scalar"

    @PYB11const
    def gamma(self,
              massDensity = "const Scalar",
              specificThermalEnergy = "const Scalar"):
        return "Scalar"

    @PYB11const
    def bulkModulus(self,
                    massDensity = "const Scalar",
                    specificThermalEnergy = "const Scalar"):
        return "Scalar"

    @PYB11const
    def entropy(self,
                massDensity = "const Scalar",
                specificThermalEnergy = "const Scalar"):
        return "Scalar"

    #...........................................................................
    # Properties
    materialNumber = PYB11property()
    numRhoVals = PYB11property()
    numTvals = PYB11property()
    rhoMin = PYB11property()
    rhoMax = PYB11property()
    Tmin = PYB11property()
    Tmax = PYB11property()
    specificThermalEnergyVals = PYB11property(getterraw="[](const ANEOS<%(Dimension)s>& self) { return ANEOS_STEvals(self); }",
                                              doc="Get the specific thermal energy lookup table values")
    atomicWeight = PYB11property("double", "atomicWeight", "atomicWeight")
    externalPressure = PYB11property("double", "externalPressure", "externalPressure")
    
#-------------------------------------------------------------------------------
# Add the virtual interface
#-------------------------------------------------------------------------------
PYB11inject(EOSAbstractMethods, ANEOS, virtual=True)
