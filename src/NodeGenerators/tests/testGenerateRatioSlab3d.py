from Spheral3d import *
from GenerateRatioSlab import *
from SpheralTestUtilities import *
from VoronoiDistributeNodes import distributeNodes3d as distributeNodes
from siloPointmeshDump import *

commandLine(hmin    = 1e-5,
            hmax    = 1e6,
            res0 = 1.0,
            xratio = 1.5,
            yratio = 1.5,
            zratio = 1.5,
            nPerh   = 2.01,
            SPH     = False,
            rho0 = 1.0)

#-------------------------------------------------------------------------------
# Material properties.
#-------------------------------------------------------------------------------
gamma = 1.4
mu = 2.0
eos = GammaLawGasMKS(gamma, mu)

#-------------------------------------------------------------------------------
# Interpolation kernels.
#-------------------------------------------------------------------------------
WT = TableKernel(BSplineKernel(), 1000)
output("WT")

#-------------------------------------------------------------------------------
# Make the NodeList.
#-------------------------------------------------------------------------------
nodes = makeFluidNodeList("nodes", eos,
                          hmin = hmin,
                          hmax = hmax,
                          nPerh = nPerh,
                          xmin = Vector.one * -100.0,
                          xmax = Vector.one *  100.0)
output("nodes")
output("nodes.hmin")
output("nodes.hmax")
output("nodes.nodesPerSmoothingScale")

#-------------------------------------------------------------------------------
# Generate them nodes.
#-------------------------------------------------------------------------------
def rhoprofile(r):
    return exp(-r*r/(rhoscale*rhoscale))

generator = GenerateRatioSlab3d(dxSurface = res0, 
                                xratio = xratio,
                                dySurface = res0, 
                                yratio = yratio,
                                dzSurface = res0, 
                                zratio = zratio,
                                rho = rho0,
                                xmin=(0.0,0.0,0.0),
                                xmax=(10.0,10.0,10.0),
                                nNodePerh = nPerh,
                                SPH = SPH,
                                flipx=True,
                                flipy=True,
                                flipz=True)
distributeNodes((nodes, generator))

#-------------------------------------------------------------------------------
# Drop a viz file for inspection.
#-------------------------------------------------------------------------------
Hfield = nodes.Hfield()
HfieldInv = SymTensorField("H inverse", nodes)
for i in xrange(nodes.numNodes):
    HfieldInv[i] = SymTensor(Hfield[i].Inverse())
vizfile = siloPointmeshDump(baseName = "ratio_slab3d_test",
                            baseDirectory = "ratio_slab3d_test",
                            fields = [nodes.massDensity(),
                                      nodes.mass(),
                                      nodes.velocity(),
                                      nodes.specificThermalEnergy(),
                                      Hfield,
                                      HfieldInv],
                            )
