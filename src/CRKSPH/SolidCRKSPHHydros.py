from SpheralModules.Spheral.CRKSPHSpace import *
from SpheralModules.Spheral.NodeSpace import *
from SpheralModules.Spheral.PhysicsSpace import *
from SpheralModules.Spheral.PhysicsSpace import *
from SpheralModules.Spheral.KernelSpace import *

from spheralDimensions import spheralDimensions
dims = spheralDimensions()

#-------------------------------------------------------------------------------
# The generic SolidCRKSPHHydro pattern.
#-------------------------------------------------------------------------------
SolidCRKSPHHydroFactoryString = """
class %(classname)s%(dim)s(SolidCRKSPHHydroBase%(dim)s):

    def __init__(self,
                 W,
                 WPi,
                 Q,
                 Wfilter = TableKernel%(dim)s(NBSplineKernel%(dim)s(7),1000),
                 filter = 0.0,
                 cfl = 0.25,
                 useVelocityMagnitudeForDt = False,
                 compatibleEnergyEvolution = True,
                 XSPH = True,
                 densityUpdate = RigorousSumDensity,
                 HUpdate = IdealH,
                 epsTensile = 0.0,
                 nTensile = 4.0):
        self._smoothingScaleMethod = %(smoothingScaleMethod)s%(dim)s()
        SolidCRKSPHHydroBase%(dim)s.__init__(self,
                                             self._smoothingScaleMethod,
                                             W,
                                             WPi,
                                             Q,
                                             Wfilter,
                                             filter,
                                             cfl,
                                             useVelocityMagnitudeForDt,
                                             compatibleEnergyEvolution,
                                             XSPH,
                                             densityUpdate,
                                             HUpdate,
                                             epsTensile,
                                             nTensile)
        return
"""

#-------------------------------------------------------------------------------
# Make 'em.
#-------------------------------------------------------------------------------
for dim in dims:
    exec(SolidCRKSPHHydroFactoryString % {"dim"                  : "%id" % dim,
                                          "classname"            : "SolidCRKSPHHydro",
                                          "smoothingScaleMethod" : "SPHSmoothingScale"})
    exec(SolidCRKSPHHydroFactoryString % {"dim"                  : "%id" % dim,
                                          "classname"            : "SolidACRKSPHHydro",
                                          "smoothingScaleMethod" : "ASPHSmoothingScale"})
