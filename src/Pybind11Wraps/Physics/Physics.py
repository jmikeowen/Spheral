#-------------------------------------------------------------------------------
# Physics base class
#-------------------------------------------------------------------------------
from PYB11Generator import *
from PhysicsAbstractMethods import *

@PYB11template("Dimension")
@PYB11module("SpheralPhysics")
class Physics:

    PYB11typedefs = """
    typedef typename %(Dimension)s::Scalar Scalar;
    typedef typename %(Dimension)s::Vector Vector;
    typedef typename %(Dimension)s::Tensor Tensor;
    typedef typename %(Dimension)s::SymTensor SymTensor;
    typedef typename %(Dimension)s::ThirdRankTensor ThirdRankTensor;
    typedef typename Physics<%(Dimension)s>::TimeStepType TimeStepType;
"""

    #...........................................................................
    # Constructors
    def pyinit(self):
        "Physics default constructor"

    #...........................................................................
    # Virtual methods
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

    @PYB11virtual
    def initializeProblemStartup(self,
                                 dataBase = "DataBase<%(Dimension)s>&"):
        "An optional hook to initialize once when the problem is starting up."
        return "void"

    @PYB11virtual
    def preStepInitialize(self,
                          dataBase = "const DataBase<%(Dimension)s>&", 
                          state = "State<%(Dimension)s>&",
                          derivs = "StateDerivatives<%(Dimension)s>&"):
        "Optional hook to be called at the beginning of a time step."
        return "void"

    @PYB11virtual
    def initialize(self,
                   time = "const Scalar", 
                   dt = "const Scalar",
                   dataBase = "const DataBase<%(Dimension)s>&", 
                   state = "State<%(Dimension)s>&",
                   derivs = "StateDerivatives<%(Dimension)s>&"):
        "Some packages might want a hook to do some initializations before the evaluateDerivatives() method is called."
        return "void"

    @PYB11virtual
    def finalize(self,
                 time = "const Scalar", 
                 dt = "const Scalar",
                 dataBase = "DataBase<%(Dimension)s>&", 
                 state = "State<%(Dimension)s>&",
                 derivs = "StateDerivatives<%(Dimension)s>&"):
        "Similarly packages might want a hook to do some post-step finalizations.  Really we should rename this post-step finalize."
        return "void"

    @PYB11virtual
    @PYB11const
    def finalizeDerivatives(self,
                            time = "const Scalar",
                            dt = "const Scalar",
                            dataBase = "const DataBase<%(Dimension)s>&",
                            state = "const State<%(Dimension)s>&",
                            derivs = "StateDerivatives<%(Dimension)s>&"):
        "Provide a hook to be called after all physics packages have had their evaluateDerivatives method called, but before anyone does anything with those derivatives."
        return "void"

    @PYB11virtual
    def postStateUpdate(self,
                        time = "const Scalar", 
                        dt = "const Scalar",
                        dataBase = "const DataBase<%(Dimension)s>&", 
                        state = "State<%(Dimension)s>&",
                        derivs = "StateDerivatives<%(Dimension)s>&"):
        "Provide a hook to be called after the state has been updated and boundary conditions have been enforced."
        return "void"

    @PYB11virtual
    @PYB11const
    def requireConnectivity(self):
        "Some physics does not require the connectivity be constructed."
        return "bool"

    @PYB11virtual
    @PYB11const
    def requireGhostConnectivity(self):
        "Some physics algorithms require ghost connectivity to be constructed."
        return "bool"

    @PYB11virtual
    @PYB11const
    def requireOverlapConnectivity(self):
        "Some physics algorithms require overlap connectivity to be constructed."
        return "bool"

    @PYB11virtual
    @PYB11const
    def requireReproducingKernels(self):
        "Some physics algorithms require reproducing kernels."
        return "std::set<RKOrder>"

    @PYB11virtual
    @PYB11const
    def requireReproducingKernelHessian(self):
        "Some physics algorithms require reproducing kernel spatial second derivatives."
        return "bool"

    @PYB11virtual
    @PYB11const
    def updateReproducingKernelsInFinalize(self):
        "Does this package need an update of reproducing kernels during finalize?"
        return "bool"
    
    @PYB11virtual
    @PYB11const
    def extraEnergy(self):
        "Many physics packages will have their own representations of energy in the system (gravitational potential energy, radiative losses, etc.)"
        return "Scalar"

    @PYB11virtual
    @PYB11const
    def extraMomentum(self):
        "Many physics packages will also have their own representations of momentum in the system (electromagnetic momentum flux density, etc.)"
        return "Vector"

    @PYB11virtual
    def registerAdditionalVisualizationState(self,
                                             dataBase = "DataBase<%(Dimension)s>&",
                                             state = "State<%(Dimension)s>&"):
        "Register any additional state for visualization."
        return "void"

    #...........................................................................
    # Methods
    def appendBoundary(self, boundary="Boundary<%(Dimension)s>&"):
        "Add a Boundary condition to the end of boundary list"
        return "void"
    
    def prependBoundary(self, boundary="Boundary<%(Dimension)s>&"):
        "Insert a Boundary condition at the beginning of the boundary list"
        return "void"

    def clearBoundaries(self):
        "Remove all boundary conditions"
        return "void"

    @PYB11const
    def haveBoundary(self, boundary="const Boundary<%(Dimension)s>&"):
        "Test if the given Boundary condition is registered."
        return "bool"

    @PYB11returnpolicy("reference_internal")
    @PYB11const
    def boundaryConditions(self):
        "Access the list of boundary conditions."
        return "const std::vector<Boundary<%(Dimension)s>*>&"

#-------------------------------------------------------------------------------
# Inject abstract interface
#-------------------------------------------------------------------------------
PYB11inject(PhysicsAbstractMethods, Physics, pure_virtual=True)
