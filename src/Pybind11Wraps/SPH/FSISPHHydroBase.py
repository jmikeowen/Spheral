#-------------------------------------------------------------------------------
# FSISPHHydroBase
#-------------------------------------------------------------------------------
from PYB11Generator import *
from SPHHydroBase import *
from RestartMethods import *

@PYB11template("Dimension")
@PYB11module("SpheralSPH")
class FSISPHHydroBase(SPHHydroBase):
    "FSISPHHydroBase -- SPHHydro modified for large density discontinuities"

    PYB11typedefs = """
  typedef typename %(Dimension)s::Scalar Scalar;
  typedef typename %(Dimension)s::Vector Vector;
  typedef typename %(Dimension)s::Tensor Tensor;
  typedef typename %(Dimension)s::SymTensor SymTensor;
  typedef typename Physics<%(Dimension)s>::TimeStepType TimeStepType;
"""
    
    def pyinit(smoothingScaleMethod = "const SmoothingScaleBase<%(Dimension)s>&",
               dataBase = "DataBase<%(Dimension)s>&",
               Q = "ArtificialViscosity<%(Dimension)s>&",
               W = "const TableKernel<%(Dimension)s>&",
               WPi = "const TableKernel<%(Dimension)s>&",
               alpha = "const double",
               diffusionCoefficient = "const double",
               sumDensityNodeLists = "std::vector<int>",
               filter = "const double",
               cfl = "const double",
               useVelocityMagnitudeForDt = "const bool",
               compatibleEnergyEvolution = "const bool",
               evolveTotalEnergy = "const bool",
               gradhCorrection = "const bool",
               XSPH = "const bool",
               correctVelocityGradient = "const bool",
               sumMassDensityOverAllNodeLists = "const bool",
               densityUpdate = "const MassDensityType",
               HUpdate = "const HEvolutionType",
               epsTensile = "const double",
               nTensile = "const double",
               xmin = "const Vector&",
               xmax = "const Vector&"):
        "FSISPHHydroBase constructor"

               
    #...........................................................................
    # Virtual methods
    @PYB11virtual
    @PYB11const
    def evaluateDerivatives(time = "const Scalar",
                            dt = "const Scalar",
                            dataBase = "const DataBase<%(Dimension)s>&",
                            state = "const State<%(Dimension)s>&",
                            derivs = "StateDerivatives<%(Dimension)s>&"):
        """Evaluate the derivatives for the principle hydro 
mass density, velocity, and specific thermal energy."""
        return "void"


    #...........................................................................
    # Properties
    alpha = PYB11property("double", "alpha", "alpha",
                           doc="exponent coefficient in Monaghans generalized momentum eqn.")
    sumDensityNodeLists = PYB11property("std::vector<int>", "sumDensityNodeLists", "sumDensityNodeLists", 
                                              doc="control if rigorous density sum is applied to individual node lists.")
    diffusionCoefficient = PYB11property("double", "diffusionCoefficient", "diffusionCoefficient", 
                                          doc="coefficient used to diffuse density and specific thermal energy amongst like nodes.")
#-------------------------------------------------------------------------------
# Inject methods
#-------------------------------------------------------------------------------
PYB11inject(RestartMethods, FSISPHHydroBase)
