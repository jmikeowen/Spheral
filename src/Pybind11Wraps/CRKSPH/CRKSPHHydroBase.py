#-------------------------------------------------------------------------------
# CRKSPHHydroBase
#-------------------------------------------------------------------------------
from PYB11Generator import *
from GenericHydro import *
from RestartMethods import *

@PYB11template("Dimension")
@PYB11module("SpheralCRKSPH")
class CRKSPHHydroBase(GenericHydro):
    "CRKSPHHydroBase -- The CRKSPH/ACRKSPH hydrodynamic package for Spheral++."

    PYB11typedefs = """
    typedef typename %(Dimension)s::Scalar Scalar;
    typedef typename %(Dimension)s::Vector Vector;
    typedef typename %(Dimension)s::Tensor Tensor;
    typedef typename %(Dimension)s::SymTensor SymTensor;
    typedef typename %(Dimension)s::ThirdRankTensor ThirdRankTensor;
    typedef typename %(Dimension)s::FourthRankTensor FourthRankTensor;
    typedef typename %(Dimension)s::FifthRankTensor FifthRankTensor;
    typedef typename Physics<%(Dimension)s>::TimeStepType TimeStepType;
"""

    def pyinit(self,
               smoothingScaleMethod = "const SmoothingScaleBase<%(Dimension)s>&",
               Q = "ArtificialViscosity<%(Dimension)s>&",
               W = "const TableKernel<%(Dimension)s>&",
               WPi = "const TableKernel<%(Dimension)s>&",
               filter = "const double",
               cfl = "const double",
               useVelocityMagnitudeForDt = "const bool",
               compatibleEnergyEvolution = "const bool",
               evolveTotalEnergy = "const bool",
               XSPH = "const bool",
               densityUpdate = "const MassDensityType",
               HUpdate = "const HEvolutionType",
               correctionOrder = "const CRKOrder",
               volumeType = "const CRKVolumeType",
               epsTensile = "const double",
               nTensile = "const double"):
        "Constructor"

    #...........................................................................
    # Virtual methods
    @PYB11virtual
    def initializeProblemStartup(self, dataBase = "DataBase<%(Dimension)s>&"):
        "Tasks we do once on problem startup."
        return "void"

    @PYB11virtual 
    def registerState(self,
                      dataBase = "DataBase<%(Dimension)s>&",
                      state = "State<%(Dimension)s>&"):
        "Register the state Hydro expects to use and evolve."
        return "void"

    @PYB11virtual
    def registerDerivatives(self,
                            dataBase = "DataBase<%(Dimension)s>&",
                            derivs = "StateDerivatives<%(Dimension)s>&"):
        "Register the derivatives/change fields for updating state."
        return "void"

    @PYB11virtual
    def initialize(self,
                   time = "const Scalar",
                   dt = "const Scalar",
                   dataBase = "const DataBase<%(Dimension)s>&",
                   state = "State<%(Dimension)s>&",
                   derivs = "StateDerivatives<%(Dimension)s>&"):
        "Initialize the Hydro before we start a derivative evaluation."
        return "void"
                          
    @PYB11virtual
    @PYB11const
    def evaluateDerivatives(self,
                            time = "const Scalar",
                            dt = "const Scalar",
                            dataBase = "const DataBase<%(Dimension)s>&",
                            state = "const State<%(Dimension)s>&",
                            derivs = "StateDerivatives<%(Dimension)s>&"):
        """Evaluate the derivatives for the principle hydro variables:
mass density, velocity, and specific thermal energy."""
        return "void"

    @PYB11virtual
    @PYB11const
    def finalizeDerivatives(self,
                            time = "const Scalar",
                            dt = "const Scalar",
                            dataBase = "const DataBase<%(Dimension)s>&",
                            state = "const State<%(Dimension)s>&",
                            derivs = "StateDerivatives<%(Dimension)s>&"):
        "Finalize the derivatives."
        return "void"

    @PYB11virtual
    def finalize(self,
                 time = "const Scalar",
                 dt = "const Scalar",
                 dataBase = "DataBase<%(Dimension)s>&",
                 state = "State<%(Dimension)s>&",
                 derivs = "StateDerivatives<%(Dimension)s>&"):
        "Finalize the hydro at the completion of an integration step."
        return "void"
                  
    @PYB11virtual
    def applyGhostBoundaries(self,
                             state = "State<%(Dimension)s>&",
                             derivs = "StateDerivatives<%(Dimension)s>&"):
        "Apply boundary conditions to the physics specific fields."
        return "void"

    @PYB11virtual
    def enforceBoundaries(self,
                          state = "State<%(Dimension)s>&",
                          derivs = "StateDerivatives<%(Dimension)s>&"):
        "Enforce boundary conditions for the physics specific fields."
        return "void"

    #...........................................................................
    # Properties
    densityUpdate = PYB11property("MassDensityType", "densityUpdate", "densityUpdate",
                                  doc="Flag to choose whether we want to sum for density, or integrate the continuity equation.")
    HEvolution = PYB11property("HEvolutionType", "HEvolution", "HEvolution",
                               doc="Flag to select how we want to evolve the H tensor.")
    correctionOrder = PYB11property("CRKOrder", "correctionOrder", "correctionOrder",
                                    doc="Flag to choose CRK Correction Order")
    volumeType = PYB11property("CRKVolumeType", "volumeType", "volumeType",
                               doc="Flag for the CRK volume weighting definition")
    compatibleEnergyEvolution = PYB11property("bool", "compatibleEnergyEvolution", "compatibleEnergyEvolution",
                                              doc="Flag to determine if we're using the total energy conserving compatible energy evolution scheme.")
    evolveTotalEnergy = PYB11property("bool", "evolveTotalEnergy", "evolveTotalEnergy",
                                      doc="Flag controlling if we evolve total or specific energy.")
    XSPH = PYB11property("bool", "XSPH", "XSPH",
                         doc="Flag to determine if we're using the XSPH algorithm.")
    smoothingScaleMethod = PYB11property("SmoothingScaleBase<%(Dimension)s>&", "smoothingScaleMethod", returnpolicy="reference_internal",
                                         doc="The object defining how we evolve smoothing scales.")
    filter = PYB11property("double", "filter", "filter",
                           doc="Fraction of centroidal filtering to apply.")
    epsilonTensile = PYB11property("Scalar", "epsilonTensile", "epsilonTensile",
                                   doc="Parameters for the tensile correction force at small scales.")
    nTensile = PYB11property("Scalar", "nTensile", "nTensile",
                                   doc="Parameters for the tensile correction force at small scales.")
    voidBoundary = PYB11property("const CRKSPHVoidBoundary<%(Dimension)s>&", "voidBoundary", returnpolicy="reference_internal",
                                 doc="We maintain a special boundary condition to handle void points.")

    timeStepMask = PYB11property("const FieldList<%(Dimension)s, int>&", "timeStepMask", returnpolicy="reference_internal")
    pressure = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "pressure", returnpolicy="reference_internal")
    soundSpeed = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "soundSpeed", returnpolicy="reference_internal")
    specificThermalEnergy0 = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "specificThermalEnergy0", returnpolicy="reference_internal")
    entropy = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "entropy", returnpolicy="reference_internal")
    Hideal = PYB11property("const FieldList<%(Dimension)s, SymTensor>&", "Hideal", returnpolicy="reference_internal")
    maxViscousPressure = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "maxViscousPressure", returnpolicy="reference_internal")
    effectiveViscousPressure = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "effectiveViscousPressure", returnpolicy="reference_internal")
    viscousWork = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "viscousWork", returnpolicy="reference_internal")
    weightedNeighborSum = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "weightedNeighborSum", returnpolicy="reference_internal")
    massSecondMoment = PYB11property("const FieldList<%(Dimension)s, SymTensor>&", "massSecondMoment", returnpolicy="reference_internal")
    volume = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "volume", returnpolicy="reference_internal")
    massDensityGradient = PYB11property("const FieldList<%(Dimension)s, Vector>&", "massDensityGradient", returnpolicy="reference_internal")
    XSPHDeltaV = PYB11property("const FieldList<%(Dimension)s, Vector>&", "XSPHDeltaV", returnpolicy="reference_internal")
    DxDt = PYB11property("const FieldList<%(Dimension)s, Vector>&", "DxDt", returnpolicy="reference_internal")

    DvDt = PYB11property("const FieldList<%(Dimension)s, Vector>&", "DvDt", returnpolicy="reference_internal")
    DmassDensityDt = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "DmassDensityDt", returnpolicy="reference_internal")
    DspecificThermalEnergyDt = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "DspecificThermalEnergyDt", returnpolicy="reference_internal")
    DHDt = PYB11property("const FieldList<%(Dimension)s, SymTensor>&", "DHDt", returnpolicy="reference_internal")
    DvDx = PYB11property("const FieldList<%(Dimension)s, Tensor>&", "DvDx", returnpolicy="reference_internal")
    internalDvDx = PYB11property("const FieldList<%(Dimension)s, Tensor>&", "internalDvDx", returnpolicy="reference_internal")
    pairAccelerations = PYB11property("const FieldList<%(Dimension)s, std::vector<Vector> >&", "pairAccelerations", returnpolicy="reference_internal")
    deltaCentroid = PYB11property("const FieldList<%(Dimension)s, Vector>&", "deltaCentroid", returnpolicy="reference_internal")

    A = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "A", returnpolicy="reference_internal")
    B = PYB11property("const FieldList<%(Dimension)s, Vector>&", "B", returnpolicy="reference_internal")
    C = PYB11property("const FieldList<%(Dimension)s, Tensor>&", "C", returnpolicy="reference_internal")
    gradA = PYB11property("const FieldList<%(Dimension)s, Vector>&", "gradA", returnpolicy="reference_internal")
    gradB = PYB11property("const FieldList<%(Dimension)s, Tensor>&", "gradB", returnpolicy="reference_internal")
    gradC = PYB11property("const FieldList<%(Dimension)s, ThirdRankTensor>&", "gradC", returnpolicy="reference_internal")
    
    m0 = PYB11property("const FieldList<%(Dimension)s, Scalar>&", "m0", returnpolicy="reference_internal")
    m1 = PYB11property("const FieldList<%(Dimension)s, Vector>&", "m1", returnpolicy="reference_internal")
    m2 = PYB11property("const FieldList<%(Dimension)s, SymTensor>&", "m2", returnpolicy="reference_internal")
    m3 = PYB11property("const FieldList<%(Dimension)s, ThirdRankTensor>&", "m3", returnpolicy="reference_internal")
    m4 = PYB11property("const FieldList<%(Dimension)s, FourthRankTensor>&", "m4", returnpolicy="reference_internal")
    gradm0 = PYB11property("const FieldList<%(Dimension)s, Vector>&", "gradm0", returnpolicy="reference_internal")
    gradm1 = PYB11property("const FieldList<%(Dimension)s, Tensor>&", "gradm1", returnpolicy="reference_internal")
    gradm2 = PYB11property("const FieldList<%(Dimension)s, ThirdRankTensor> &", "gradm2", returnpolicy="reference_internal")
    gradm3 = PYB11property("const FieldList<%(Dimension)s, FourthRankTensor>&", "gradm3", returnpolicy="reference_internal")
    gradm4 = PYB11property("const FieldList<%(Dimension)s, FifthRankTensor>&", "gradm4", returnpolicy="reference_internal")

    surfacePoint = PYB11property("const FieldList<%(Dimension)s, int>&", "surfacePoint", returnpolicy="reference_internal")
    etaVoidPoints = PYB11property("const FieldList<%(Dimension)s, std::vector<Vector>>&", "etaVoidPoints", returnpolicy="reference_internal")

#-------------------------------------------------------------------------------
# Inject methods
#-------------------------------------------------------------------------------
PYB11inject(RestartMethods, CRKSPHHydroBase)

