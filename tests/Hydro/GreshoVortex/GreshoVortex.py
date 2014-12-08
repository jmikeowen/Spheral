#-------------------------------------------------------------------------------
# The Gresho-Vortex Test
#-------------------------------------------------------------------------------
import shutil
from math import *
from Spheral2d import *
from SpheralTestUtilities import *
from SpheralGnuPlotUtilities import *
from findLastRestart import *
from GenerateNodeDistribution2d import *
from CubicNodeGenerator import GenerateSquareNodeDistribution
from CentroidalVoronoiRelaxation import *

import mpi
import DistributeNodes

title("2-D integrated hydro test --  Gresho-Vortex Test")

#-------------------------------------------------------------------------------
# Generic problem parameters
#-------------------------------------------------------------------------------
commandLine(
    rho = 1.0,
    gamma = 5.0/3.0,

    # Translation
    velTx=0.0,
    velTy=0.0,

    # Geometry of Box
    x0 = -0.5,
    x1 =  0.5,
    y0 = -0.5,
    y1 =  0.5,
   
    #Center of Vortex
    xc=0.0,
    yc=0.0,

    # Resolution and node seeding.
    nx1 = 64,
    ny1 = 64,
    seed = "lattice",

    nPerh = 1.51,

    SVPH = False,
    CSPH = False,
    ASPH = False,
    SPH = True,   # This just chooses the H algorithm -- you can use this with CSPH for instance.
    filter = 0.0,  # For CSPH
    momentumConserving = True, # For CSPH
    KernelConstructor = BSplineKernel,
    Qconstructor = MonaghanGingoldViscosity,
    #Qconstructor = TensorMonaghanGingoldViscosity,
    boolReduceViscosity = False,
    nh = 5.0,
    aMin = 0.1,
    aMax = 2.0,
    linearConsistent = False,
    fcentroidal = 0.0,
    fcellPressure = 0.0,
    Cl = 1.0, 
    Cq = 0.75,
    Qlimiter = False,
    balsaraCorrection = False,
    epsilon2 = 1e-2,
    hmin = 1e-5,
    hmax = 0.5,
    hminratio = 0.1,
    cfl = 0.5,
    XSPH = False,
    epsilonTensile = 0.0,
    nTensile = 8,

    IntegratorConstructor = CheapSynchronousRK2Integrator,
    goalTime = 1.0,
    steps = None,
    vizCycle = 20,
    vizTime = 0.1,
    dt = 0.0001,
    dtMin = 1.0e-5, 
    dtMax = 0.1,
    dtGrowth = 2.0,
    maxSteps = None,
    statsStep = 10,
    HUpdate = IdealH,
    domainIndependent = False,
    rigorousBoundaries = False,
    dtverbose = False,

    densityUpdate = RigorousSumDensity, # VolumeScaledDensity,
    compatibleEnergy = True,
    gradhCorrection = False,

    useVoronoiOutput = False,
    clearDirectories = False,
    restoreCycle = None,
    restartStep = 200,
    dataDir = "dumps-greshovortex-xy",
    graphics = True,
    smooth = None,
    )

# Decide on our hydro algorithm.
if SVPH:
    if ASPH:
        HydroConstructor = ASVPHFacetedHydro
    else:
        HydroConstructor = SVPHFacetedHydro
elif CSPH:
    if ASPH:
        HydroConstructor = ACSPHHydro
    else:
        HydroConstructor = CSPHHydro
else:
    if ASPH:
        HydroConstructor = ASPHHydro
    else:
        HydroConstructor = SPHHydro

# Build our directory paths.
densityUpdateLabel = {IntegrateDensity : "IntegrateDensity",
                      SumDensity : "SumDensity",
                      RigorousSumDensity : "RigorousSumDensity",
                      SumVoronoiCellDensity : "SumVoronoiCellDensity"}
baseDir = os.path.join(dataDir,
                       HydroConstructor.__name__,
                       Qconstructor.__name__,
                       KernelConstructor.__name__,
                       "Cl=%g_Cq=%g" % (Cl, Cq),
                       densityUpdateLabel[densityUpdate],
                       "compatibleEnergy=%s" % compatibleEnergy,
                       "XSPH=%s" % XSPH,
                       "nPerh=%3.1f" % nPerh,
                       "fcentroidal=%1.3f" % max(fcentroidal, filter),
                       "fcellPressure = %1.3f" % fcellPressure,
                       "%ix%i" % (nx1, ny1))
restartDir = os.path.join(baseDir, "restarts")
restartBaseName = os.path.join(restartDir, "greshovortex-xy-%ix%i" % (nx1, ny1))

vizDir = os.path.join(baseDir, "visit")
if vizTime is None and vizCycle is None:
    vizBaseName = None
else:
    vizBaseName = "greshovortex-xy-%ix%i" % (nx1, ny1)

#-------------------------------------------------------------------------------
# Check if the necessary output directories exist.  If not, create them.
#-------------------------------------------------------------------------------
import os, sys
if mpi.rank == 0:
    if clearDirectories and os.path.exists(baseDir):
        shutil.rmtree(baseDir)
    if not os.path.exists(restartDir):
        os.makedirs(restartDir)
    if not os.path.exists(vizDir):
        os.makedirs(vizDir)
mpi.barrier()

#-------------------------------------------------------------------------------
# If we're restarting, find the set of most recent restart files.
#-------------------------------------------------------------------------------
if restoreCycle is None:
    restoreCycle = findLastRestart(restartBaseName)

#-------------------------------------------------------------------------------
# Material properties.
#-------------------------------------------------------------------------------
mu = 1.0
eos = GammaLawGasMKS(gamma, mu)

#-------------------------------------------------------------------------------
# Interpolation kernels.
#-------------------------------------------------------------------------------
WT = TableKernel(KernelConstructor(), 1000)
WTPi = WT # TableKernel(HatKernel(1.0, 1.0), 1000)
output("WT")
output("WTPi")
kernelExtent = WT.kernelExtent

#-------------------------------------------------------------------------------
# Make the NodeLists.
#-------------------------------------------------------------------------------
nodes = makeFluidNodeList("fluid", eos,
                               hmin = hmin,
                               hmax = hmax,
                               hminratio = hminratio,
                               nPerh = nPerh)
output("nodes.name")
output("    nodes.hmin")
output("    nodes.hmax")
output("    nodes.hminratio")
output("    nodes.nodesPerSmoothingScale")

#-------------------------------------------------------------------------------
# Set the node properties.
#-------------------------------------------------------------------------------
if restoreCycle is None:
    rmin = 0.0
    rmax = sqrt(2.0)*(x1-x0)
    
    if(seed=="latticeCylindrical"):
        rmin = x1-8.0*nPerh/nx1
        rmax = x1-2.0*nPerh/nx1
    
    generator = GenerateNodeDistribution2d(nx1, ny1, rho,
                                           distributionType = seed,
                                           xmin = (x0, y0),
                                           xmax = (x1, y1),
                                           #rmin = 0.0,
                                           theta = 2.0*pi,
                                           #rmax = sqrt(2.0)*(x1 - x0),
                                           rmax = rmax,
                                           rmin = rmin,
                                           nNodePerh = nPerh,
                                           SPH = SPH)

    if mpi.procs > 1:
        from VoronoiDistributeNodes import distributeNodes2d
    else:
        from DistributeNodes import distributeNodes2d

    distributeNodes2d((nodes, generator))
    print nodes.name, ":"
    output("    mpi.reduce(nodes.numInternalNodes, mpi.MIN)")
    output("    mpi.reduce(nodes.numInternalNodes, mpi.MAX)")
    output("    mpi.reduce(nodes.numInternalNodes, mpi.SUM)")

    #Set IC
    vel = nodes.velocity()
    eps = nodes.specificThermalEnergy()
    pos = nodes.positions()
    for i in xrange(nodes.numInternalNodes):
        xi, yi = pos[i]
        r2=(xi-xc)*(xi-xc)+(yi-yc)*(yi-yc)
        ri=sqrt(r2)
        vphi=0.0
        sinPhi=(yi-yc)/ri
        cosPhi=(xi-xc)/ri
        Pi=0.0
        if ri < 0.2:
           Pi=5.0+12.5*r2
 	   vphi=5*ri
        elif ri < 0.4 and ri >= 0.2:
           Pi=9.0+12.5*r2-20.0*ri+4.0*log(5.0*ri)
	   vphi=2.0-5.0*ri
        else:
           Pi=3.0+4*log(2.0)
           #vphi is zero
        velx=velTx-vphi*sinPhi #translation velocity + azimuthal velocity 
        vely=velTy+vphi*cosPhi
        vel[i]=Vector(velx,vely)
        eps0 = Pi/((gamma - 1.0)*rho)
        eps[i]=eps0

#-------------------------------------------------------------------------------
# Construct a DataBase to hold our node lists
#-------------------------------------------------------------------------------
db = DataBase()
output("db")
db.appendNodeList(nodes)
output("db.numNodeLists")
output("db.numFluidNodeLists")

#-------------------------------------------------------------------------------
# Construct the artificial viscosity.
#-------------------------------------------------------------------------------
q = Qconstructor(Cl, Cq)
q.epsilon2 = epsilon2
q.limiter = Qlimiter
q.balsaraShearCorrection = balsaraCorrection
output("q")
output("q.Cl")
output("q.Cq")
output("q.epsilon2")
output("q.limiter")
output("q.balsaraShearCorrection")

#-------------------------------------------------------------------------------
# Construct the hydro physics object.
#-------------------------------------------------------------------------------
if SVPH:
    hydro = HydroConstructor(WT, q,
                             cfl = cfl,
                             compatibleEnergyEvolution = compatibleEnergy,
                             densityUpdate = densityUpdate,
                             XSVPH = XSPH,
                             linearConsistent = linearConsistent,
                             generateVoid = False,
                             HUpdate = HUpdate,
                             fcentroidal = fcentroidal,
                             fcellPressure = fcellPressure,
                             xmin = Vector(x0 - (x1 - x0), y0 - (y1 - y0)),
                             xmax = Vector(x1 + (x1 - x0), y3 + (y1 - y0)))
elif CSPH:
    hydro = HydroConstructor(WT, WTPi, q,
                             filter = filter,
                             epsTensile = epsilonTensile,
                             nTensile = nTensile,
                             cfl = cfl,
                             compatibleEnergyEvolution = compatibleEnergy,
                             XSPH = XSPH,
                             densityUpdate = densityUpdate,
                             HUpdate = HUpdate,
                             momentumConserving = momentumConserving)
else:
    hydro = HydroConstructor(WT,
                             WTPi,
                             q,
                             cfl = cfl,
                             compatibleEnergyEvolution = compatibleEnergy,
                             gradhCorrection = gradhCorrection,
                             XSPH = XSPH,
                             densityUpdate = densityUpdate,
                             HUpdate = HUpdate,
                             epsTensile = epsilonTensile,
                             nTensile = nTensile)
output("hydro")
output("hydro.kernel()")
output("hydro.PiKernel()")
output("hydro.cfl")
output("hydro.compatibleEnergyEvolution")
output("hydro.densityUpdate")
output("hydro.HEvolution")

packages = [hydro]

#-------------------------------------------------------------------------------
# Construct the MMRV physics object.
#-------------------------------------------------------------------------------

if boolReduceViscosity:
    #q.reducingViscosityCorrection = True
    evolveReducingViscosityMultiplier = MorrisMonaghanReducingViscosity(q,nh,aMin,aMax)
    
    packages.append(evolveReducingViscosityMultiplier)


#-------------------------------------------------------------------------------
# Create boundary conditions.
#-------------------------------------------------------------------------------
xPlane0 = Plane(Vector(x0, y0), Vector( 1.0,  0.0))
xPlane1 = Plane(Vector(x1, y0), Vector(-1.0,  0.0))
yPlane0 = Plane(Vector(x0, y0), Vector( 0.0,  1.0))
yPlane1 = Plane(Vector(x0, y1), Vector( 0.0, -1.0))

xbc = PeriodicBoundary(xPlane0, xPlane1)
ybc = PeriodicBoundary(yPlane0, yPlane1)

xbc0 = ReflectingBoundary(xPlane0)
xbc1 = ReflectingBoundary(xPlane1)
ybc0 = ReflectingBoundary(yPlane0)
ybc1 = ReflectingBoundary(yPlane1)

bcSet = [xbc, ybc]
#bcSet = [xbc0, xbc1, ybc0, ybc1]

for p in packages:
    for bc in bcSet:
        p.appendBoundary(bc)

#-------------------------------------------------------------------------------
# Construct a time integrator, and add the physics packages.
#-------------------------------------------------------------------------------
integrator = IntegratorConstructor(db)
for p in packages:
    integrator.appendPhysicsPackage(p)
integrator.cullGhostNodes = False
integrator.lastDt = dt
integrator.dtMin = dtMin
integrator.dtMax = dtMax
integrator.dtGrowth = dtGrowth
integrator.domainDecompositionIndependent = domainIndependent
integrator.verbose = dtverbose
integrator.rigorousBoundaries = rigorousBoundaries
output("integrator")
output("integrator.havePhysicsPackage(hydro)")
output("integrator.lastDt")
output("integrator.dtMin")
output("integrator.dtMax")
output("integrator.dtGrowth")
output("integrator.domainDecompositionIndependent")
output("integrator.rigorousBoundaries")
output("integrator.verbose")

#-------------------------------------------------------------------------------
# If requested, smooth the initial conditions.
#-------------------------------------------------------------------------------
if smooth:
    for iter in xrange(smooth):
        db.updateConnectivityMap(False)
        cm = db.connectivityMap()
        position_fl = db.fluidPosition
        weight_fl = db.fluidMass
        H_fl = db.fluidHfield
        m0_fl = db.newFluidScalarFieldList(0.0, "m0")
        m1_fl = db.newFluidVectorFieldList(Vector.zero, "m1")
        m2_fl = db.newFluidSymTensorFieldList(SymTensor.zero, "m2")
        A0_fl = db.newFluidScalarFieldList(0.0, "A0")
        A_fl = db.newFluidScalarFieldList(0.0, "A")
        B_fl = db.newFluidVectorFieldList(Vector.zero, "B")
        C_fl = db.newFluidVectorFieldList(Vector.zero, "C")
        D_fl = db.newFluidTensorFieldList(Tensor.zero, "D")
        gradA0_fl = db.newFluidVectorFieldList(Vector.zero, "gradA0")
        gradA_fl = db.newFluidVectorFieldList(Vector.zero, "gradA")
        gradB_fl = db.newFluidTensorFieldList(Tensor.zero, "gradB")
        computeCSPHCorrections(cm, WT, weight_fl, position_fl, H_fl, True,
                               m0_fl, m1_fl, m2_fl,
                               A0_fl, A_fl, B_fl, C_fl, D_fl, gradA0_fl, gradA_fl, gradB_fl)
        eps0 = db.fluidSpecificThermalEnergy
        vel0 = db.fluidVelocity
        eps1 = interpolateCSPH(eps0, position_fl, weight_fl, H_fl, True, 
                               A_fl, B_fl, cm, WT)
        vel1 = interpolateCSPH(vel0, position_fl, weight_fl, H_fl, True, 
                               A_fl, B_fl, cm, WT)
        eps0.assignFields(eps1)
        vel0.assignFields(vel1)

#-------------------------------------------------------------------------------
# Make the problem controller.
#-------------------------------------------------------------------------------
if useVoronoiOutput:
    import SpheralVoronoiSiloDump
    vizMethod = SpheralVoronoiSiloDump.dumpPhysicsState
else:
    import SpheralVisitDump
    vizMethod = SpheralVisitDump.dumpPhysicsState
control = SpheralController(integrator, WT,
                            statsStep = statsStep,
                            restartStep = restartStep,
                            restartBaseName = restartBaseName,
                            restoreCycle = restoreCycle,
                            vizMethod = vizMethod,
                            vizBaseName = vizBaseName,
                            vizDir = vizDir,
                            vizStep = vizCycle,
                            vizTime = vizTime,
                            skipInitialPeriodicWork = (HydroConstructor in (SVPHFacetedHydro, ASVPHFacetedHydro)),
                            SPH = SPH)
output("control")

#-------------------------------------------------------------------------------
# Advance to the end time.
#-------------------------------------------------------------------------------
if not steps is None:
    control.step(steps)

else:
    control.advance(goalTime, maxSteps)
    control.updateViz(control.totalSteps, integrator.currentTime, 0.0)
    control.dropRestartFile()

# Plot the final velocity profile
if graphics:
    pos = nodes.positions()
    vel = nodes.velocity()
    vaz = db.newFluidScalarFieldList(0.0, "azimuthal velocity")
    for i in xrange(nodes.numInternalNodes):
        rhat = (pos[i] - Vector(xc, yc)).unitVector()
        vaz[0][i] = (vel[i] - vel[i].dot(rhat)*rhat).magnitude()
    p = plotFieldList(vaz, xFunction="(%%s - Vector2d(%g,%g)).magnitude()" % (xc, yc), plotStyle="points", lineTitle="Simulation", winTitle="Velocity")
    #p = plotFieldList(db.fluidVelocity, xFunction="(%%s - Vector2d(%g,%g)).magnitude()" % (xc, yc), yFunction="%s.magnitude()", plotStyle="points", winTitle="Velocity")

    # Plot the analytic answer.
    xans = [0.0, 0.1, 0.2, 0.4, 1.0]
    yans = [0.0, 0.5, 1.0, 0.0, 0.0]
    ansData = Gnuplot.Data(xans, yans, title="Analytic", with_="lines lt 1 lw 3")
    p.replot(ansData)
