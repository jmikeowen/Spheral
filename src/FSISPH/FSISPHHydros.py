from SpheralCompiledPackages import *

from spheralDimensions import spheralDimensions
dims = spheralDimensions()

#-------------------------------------------------------------------------------
# The generic FSISPHHydro pattern.
#-------------------------------------------------------------------------------
FSISPHHydroFactoryString = """
class %(classname)s%(dim)s(FSISPHHydroBase%(dim)s):

    def __init__(self,
                 dataBase,
                 Q,
                 slides,
                 W,
                 filter = 0.0,
                 cfl = 0.5,
                 surfaceForceCoefficient = 0.0,
                 densityStabilizationCoefficient = 0.0,
                 densityDiffusionCoefficient = 0.0,   
                 specificThermalEnergyDiffusionCoefficient = 0.0,                     
                 sumDensityNodeLists = None,
                 useVelocityMagnitudeForDt = False,
                 compatibleEnergyEvolution = True,
                 evolveTotalEnergy = False,
                 gradhCorrection = False,
                 XSPH = False,
                 correctVelocityGradient = True,
                 densityUpdate = RigorousSumDensity,
                 HUpdate = IdealH,
                 epsTensile = 0.0,
                 nTensile = 4.0,
                 xmin = Vector%(dim)s(-1e100, -1e100, -1e100),
                 xmax = Vector%(dim)s( 1e100,  1e100,  1e100)):
        self._smoothingScaleMethod = %(smoothingScaleMethod)s%(dim)s()

        FSISPHHydroBase%(dim)s.__init__(self,
                                          self._smoothingScaleMethod,
                                          dataBase,
                                          Q,
                                          slides,
                                          W,
                                          filter,
                                          cfl,
                                          surfaceForceCoefficient,
                                          densityStabilizationCoefficient,
                                          densityDiffusionCoefficient,
                                          specificThermalEnergyDiffusionCoefficient,             
                                          sumDensityNodeLists,
                                          useVelocityMagnitudeForDt,
                                          compatibleEnergyEvolution,
                                          evolveTotalEnergy,
                                          gradhCorrection,
                                          XSPH,
                                          correctVelocityGradient,
                                          densityUpdate,
                                          HUpdate,
                                          epsTensile,
                                          nTensile,
                                          xmin,
                                          xmax)
        return
"""

#-------------------------------------------------------------------------------
# The generic FSISPHHydro RZ pattern.
#-------------------------------------------------------------------------------
FSISPHHydroRZFactoryString = """
class %(classname)s(FSISPHHydroBaseRZ):

    def __init__(self,
                 dataBase,
                 Q,
                 slides,
                 W,
                 filter = 0.0,
                 cfl = 0.5,
                 surfaceForceCoefficient = 0.0,
                 densityStabilizationCoefficient = 0.0,
                 densityDiffusionCoefficient = 0.0,   
                 specificThermalEnergyDiffusionCoefficient = 0.0,          
                 sumDensityNodeLists = None,
                 useVelocityMagnitudeForDt = False,
                 compatibleEnergyEvolution = True,
                 evolveTotalEnergy = False,
                 gradhCorrection = False,
                 XSPH = False,
                 correctVelocityGradient = True,
                 densityUpdate = RigorousSumDensity,
                 HUpdate = IdealH,
                 epsTensile = 0.0,
                 nTensile = 4.0,
                 xmin = Vector2d(-1e100, -1e100),
                 xmax = Vector2d( 1e100,  1e100),
                 etaMinAxis = 0.1):
        self._smoothingScaleMethod = %(smoothingScaleMethod)s2d()
        
        WPi = W

        FSISPHHydroBaseRZ.__init__(self,
                                   self._smoothingScaleMethod,
                                   dataBase,
                                    Q,
                                    slides,
                                    W,
                                    filter,
                                    cfl,
                                    surfaceForceCoefficient,
                                    densityStabilizationCoefficient,
                                    densityDiffusionCoefficient,
                                    specificThermalEnergyDiffusionCoefficient,       
                                    sumDensityNodeLists,
                                    useVelocityMagnitudeForDt,
                                    compatibleEnergyEvolution,
                                    evolveTotalEnergy,
                                    gradhCorrection,
                                    XSPH,
                                    correctVelocityGradient,
                                    densityUpdate,
                                    HUpdate,
                                    epsTensile,
                                    nTensile,
                                    xmin,
                                    xmax)
        self.zaxisBC = AxisBoundaryRZ(etaMinAxis)
        self.appendBoundary(self.zaxisBC)
        return
"""
#-------------------------------------------------------------------------------
# The generic FSISPHHydro RZ pattern.
#-------------------------------------------------------------------------------
SolidFSISPHHydroRZFactoryString = """
class %(classname)s(SolidFSISPHHydroBaseRZ):

    def __init__(self,
                 dataBase,
                 Q,
                 slides,
                 W,
                 filter = 0.0,
                 cfl = 0.5,
                 surfaceForceCoefficient = 0.0,
                 densityStabilizationCoefficient = 0.0,
                 densityDiffusionCoefficient = 0.0,   
                 specificThermalEnergyDiffusionCoefficient = 0.0,          
                 sumDensityNodeLists = None,
                 useVelocityMagnitudeForDt = False,
                 compatibleEnergyEvolution = True,
                 evolveTotalEnergy = False,
                 gradhCorrection = False,
                 XSPH = False,
                 correctVelocityGradient = True,
                 densityUpdate = RigorousSumDensity,
                 HUpdate = IdealH,
                 epsTensile = 0.0,
                 nTensile = 4.0,
                 damageRelieveRubble = False,
                 negativePressureInDamage = False,
                 strengthInDamage = False,
                 xmin = Vector2d(-1e100, -1e100),
                 xmax = Vector2d( 1e100,  1e100),
                 etaMinAxis = 0.1):
        self._smoothingScaleMethod = %(smoothingScaleMethod)s2d()

        SolidFSISPHHydroBaseRZ.__init__(self,
                                        self._smoothingScaleMethod,
                                        dataBase,
                                        Q,
                                        slides,
                                        W,
                                        filter,
                                        cfl,
                                        surfaceForceCoefficient,
                                        densityStabilizationCoefficient,
                                        densityDiffusionCoefficient,   
                                        specificThermalEnergyDiffusionCoefficient,          
                                        sumDensityNodeLists,
                                        useVelocityMagnitudeForDt,
                                        compatibleEnergyEvolution,
                                        evolveTotalEnergy,
                                        gradhCorrection,
                                        XSPH,
                                        correctVelocityGradient,
                                        densityUpdate,
                                        HUpdate,
                                        epsTensile,
                                        nTensile,
                                        damageRelieveRubble,
                                        negativePressureInDamage,
                                        strengthInDamage,
                                        xmin,
                                        xmax)
        self.zaxisBC = AxisBoundaryRZ(etaMinAxis)
        self.appendBoundary(self.zaxisBC)
        return
"""

#-------------------------------------------------------------------------------
# The generic solidFSISPHHydro pattern.
#-------------------------------------------------------------------------------
SolidFSISPHHydroFactoryString = """
class %(classname)s%(dim)s(SolidFSISPHHydroBase%(dim)s):

    def __init__(self,
                 dataBase,
                 Q,
                 slides,
                 W,
                 filter = 0.0,
                 cfl = 0.5,
                 surfaceForceCoefficient = 0.0,
                 densityStabilizationCoefficient = 0.0,
                 densityDiffusionCoefficient = 0.0,   
                 specificThermalEnergyDiffusionCoefficient = 0.0,        
                 sumDensityNodeLists = None,
                 useVelocityMagnitudeForDt = False,
                 compatibleEnergyEvolution = True,
                 evolveTotalEnergy = False,
                 gradhCorrection = False,
                 XSPH = True,
                 correctVelocityGradient = True,
                 densityUpdate = RigorousSumDensity,
                 HUpdate = IdealH,
                 epsTensile = 0.0,
                 nTensile = 4.0,
                 damageRelieveRubble = False,
                 negativePressureInDamage = False,
                 strengthInDamage = False,
                 xmin = Vector%(dim)s(-1e100, -1e100, -1e100),
                 xmax = Vector%(dim)s( 1e100,  1e100,  1e100)):
        self._smoothingScaleMethod = %(smoothingScaleMethod)s%(dim)s()

        SolidFSISPHHydroBase%(dim)s.__init__(self,
                                          self._smoothingScaleMethod,
                                          dataBase,
                                          Q,
                                          slides,
                                          W,
                                          filter,
                                          cfl,
                                          surfaceForceCoefficient,
                                          densityStabilizationCoefficient,
                                          densityDiffusionCoefficient,   
                                          specificThermalEnergyDiffusionCoefficient,       
                                          sumDensityNodeLists,
                                          useVelocityMagnitudeForDt,
                                          compatibleEnergyEvolution,
                                          evolveTotalEnergy,
                                          gradhCorrection,
                                          XSPH,
                                          correctVelocityGradient,
                                          densityUpdate,
                                          HUpdate,
                                          epsTensile,
                                          nTensile,
                                          damageRelieveRubble,
                                          negativePressureInDamage,
                                          strengthInDamage,
                                          xmin,
                                          xmax)
        return
"""

for dim in dims:
    '''
    exec(FSISPHHydroFactoryString % {"dim"                  : "%id" % dim,
                                     "classname"            : "FSISPHHydro",
                                     "smoothingScaleMethod" : "SPHSmoothingScale"})
    exec(FSISPHHydroFactoryString % {"dim"                  : "%id" % dim,
                                     "classname"            : "AFSISPHHydro",
                                     "smoothingScaleMethod" : "ASPHSmoothingScale"})
    '''
    exec(SolidFSISPHHydroFactoryString % {"dim"                  : "%id" % dim,
                                       "classname"            : "SolidFSISPHHydro",
                                       "smoothingScaleMethod" : "SPHSmoothingScale"})
    exec(SolidFSISPHHydroFactoryString % {"dim"                  : "%id" % dim,
                                       "classname"            : "SolidASISPHHydro",
                                       "smoothingScaleMethod" : "ASPHSmoothingScale"})
'''
if 2 in dims:
    exec(SolidFSISPHHydroRZFactoryString % {"classname"            : "SolidFSISPHHydroRZ",
                                             "smoothingScaleMethod" : "SPHSmoothingScale"})

    exec(SolidFSISPHHydroRZFactoryString % {"classname"            : "SolidAFSISPHHydroRZ",
                                            "smoothingScaleMethod" : "ASPHSmoothingScale"})

    exec(FSISPHHydroRZFactoryString % {"classname"            : "FSISPHHydroRZ",
                                        "smoothingScaleMethod" : "SPHSmoothingScale"})

    exec(FSISPHHydroRZFactoryString % {"classname"            : "AFSISPHHydroRZ",
                                        "smoothingScaleMethod" : "ASPHSmoothingScale"})
'''
def FSISPH(dataBase,
        W,
        Q = None,
        slides=None,
        filter = 0.0,
        cfl = 0.25,
        surfaceForceCoefficient=0.0,
        densityStabilizationCoefficient=0.0, 
        densityDiffusionCoefficient=0.0,      
        specificThermalEnergyDiffusionCoefficient=0.0,        
        sumDensityNodeLists=[],
        useVelocityMagnitudeForDt = False,
        compatibleEnergyEvolution = False,
        evolveTotalEnergy = False,
        gradhCorrection = False,
        XSPH = False,
        correctVelocityGradient = False,    # will break consistency between DrhoDt and DepsDt
        densityUpdate = IntegrateDensity,
        HUpdate = IdealH,
        epsTensile = 0.0,
        nTensile = 4.0,
        damageRelieveRubble = False,
        negativePressureInDamage = False,
        strengthInDamage = False,
        xmin = (-1e100, -1e100, -1e100),
        xmax = ( 1e100,  1e100,  1e100),
        ASPH = False,
        RZ = False):

    
    if densityDiffusionCoefficient > 1e-30 or specificThermalEnergyDiffusionCoefficient > 1e-30:
        print("**********************************************************************")
        print(" FSISPH WARNING : compatibility issue w/ StrainPorosity when running:")
        print("                  densityDiffusionCoefficient > 0.0 or")
        print("                  specificThermalEnergyDiffusionCoefficient > 0.0.")
        print("                  ")
        print("                  densityStabilizationCoefficient > 0.0 is safe")
        print("**********************************************************************")

    if gradhCorrection:
        raise RuntimeError, "gradhCorrection not implemented yet"

    #if XSPH  and dataBase.numNodeLists>1:
    #    raise RuntimeError, "XSPH is not set up for multiple nodeLists"

    if strengthInDamage and damageRelieveRubble:
        raise RuntimeError, "strengthInDamage and damageRelieveRubble are incompatible"

    # create the map nodelist --> index
    nodeLists = dataBase.nodeLists()
    nodeListMap = {}
    for i in range(dataBase.numNodeLists):          
        nodeListMap[nodeLists[i]]=i

    # for now use int 1/0 to indicate sum density or not
    sumDensitySwitch = [0]*dataBase.numNodeLists
    for nodeList in sumDensityNodeLists:
        i = nodeListMap[nodeList]
        sumDensitySwitch[i]=1
    sumDensitySwitch = vector_of_int(sumDensitySwitch)
            
    # We use the provided DataBase to sniff out what sort of NodeLists are being
    # used, and based on this determine which SPH object to build.
    ndim = dataBase.nDim
    nfluid = dataBase.numFluidNodeLists
    nsolid = dataBase.numSolidNodeLists
    if nsolid > 0 and nsolid != nfluid:
        print "SPH Error: you have provided both solid and fluid NodeLists, which is currently not supported."
        print "           If you want some fluids active, provide SolidNodeList without a strength option specfied,"
        print "           which will result in fluid behaviour for those nodes."
        raise RuntimeError, "Cannot mix solid and fluid NodeLists."

    # Decide on the hydro object.
    if RZ:

        # RZ ----------------------------------------
        if nsolid > 0:
            if ASPH:
                Constructor = SolidAFSISPHHydroRZ
            else:
                Constructor = SolidFSISPHHydroRZ
        else:
            if ASPH:
                Constructor = AFSISPHHydroRZ
            else:
                Constructor = FSISPHHydroRZ

    else:
        # Cartesian ---------------------------------
        if nsolid > 0:
            if ASPH:
                Constructor = eval("SolidAFSISPHHydro%id" % ndim)
            else:
                Constructor = eval("SolidFSISPHHydro%id" % ndim)
        else:
            if ASPH:
                Constructor = eval("AFSISPHHydro%id" % ndim)
            else:
                Constructor = eval("FSISPHHydro%id" % ndim)

    # Artificial viscosity.
    if not Q:
        Cl = 1.0*(dataBase.maxKernelExtent/2.0)
        Cq = 1.0*(dataBase.maxKernelExtent/2.0)**2
        Q = eval("MonaghanGingoldViscosity%id(Clinear=%g, Cquadratic=%g)" % (ndim, Cl, Cq))

    # slide surfaces.
    if not slides:
        contactTypes = vector_of_int([0]*(dataBase.numNodeLists**2))
        slides = eval("SlideSurface%id(W,contactTypes)" % (ndim))

    # Build the constructor arguments
    xmin = (ndim,) + xmin
    xmax = (ndim,) + xmax
    kwargs = {"dataBase" : dataBase,
              "Q" : Q,
              "slides" : slides,
              "W" : W,
              "filter" : filter,
              "cfl" : cfl,
              "surfaceForceCoefficient" : surfaceForceCoefficient,
              "densityStabilizationCoefficient" : densityStabilizationCoefficient,
              "densityDiffusionCoefficient" : densityDiffusionCoefficient,        
              "specificThermalEnergyDiffusionCoefficient" : specificThermalEnergyDiffusionCoefficient,        
              "sumDensityNodeLists" : sumDensitySwitch,
              "useVelocityMagnitudeForDt" : useVelocityMagnitudeForDt,
              "compatibleEnergyEvolution" : compatibleEnergyEvolution,
              "evolveTotalEnergy" : evolveTotalEnergy,
              "gradhCorrection" : gradhCorrection,
              "XSPH" : XSPH,
              "correctVelocityGradient" : correctVelocityGradient,
              "densityUpdate" : densityUpdate,
              "HUpdate" : HUpdate,
              "epsTensile" : epsTensile,
              "nTensile" : nTensile,
              "xmin" : eval("Vector%id(%g, %g, %g)" % xmin),
              "xmax" : eval("Vector%id(%g, %g, %g)" % xmax)}

    if nsolid > 0:
        kwargs.update({"damageRelieveRubble"      : damageRelieveRubble,
                       "negativePressureInDamage" : negativePressureInDamage,
                       "strengthInDamage"         : strengthInDamage})

    # Build and return the thing.
    result = Constructor(**kwargs)
    result.Q = Q
    result.slides = slides
    
    return result

#-------------------------------------------------------------------------------
# Provide a shorthand ASPH factory.
#-------------------------------------------------------------------------------
def AFSISPH(dataBase,
         W,
         Q = None,
         slides=None,
         filter = 0.0,
         cfl = 0.25,
         surfaceForceCoefficient = 0.0,
         densityStabilizationCoefficient = 0.0,
         densityDiffusionCoefficient = 0.0,   
         specificThermalEnergyDiffusionCoefficient = 0.0,                     
         sumDensityNodeLists = None,       
         useVelocityMagnitudeForDt = False,
         compatibleEnergyEvolution = True,
         evolveTotalEnergy = False,
         gradhCorrection = False,
         XSPH = False,
         correctVelocityGradient = True,
         densityUpdate = IntegrateDensity,
         HUpdate = IdealH,
         epsTensile = 0.0,
         nTensile = 4.0,
         damageRelieveRubble = False,
         negativePressureInDamage = False,
         strengthInDamage = False,
         xmin = (-1e100, -1e100, -1e100),
         xmax = ( 1e100,  1e100,  1e100)):
    return FSISPH(dataBase = dataBase,
               W = W,
               Q = Q,
               slides=None,
               filter = filter,
               cfl = cfl,
               surfaceForceCoefficient = 0.0,
               densityStabilizationCoefficient = 0.0,
               densityDiffusionCoefficient = 0.0,   
               specificThermalEnergyDiffusionCoefficient = 0.0,                     
               sumDensityNodeLists = None,          
               useVelocityMagnitudeForDt = useVelocityMagnitudeForDt,
               compatibleEnergyEvolution = compatibleEnergyEvolution,
               evolveTotalEnergy = evolveTotalEnergy,
               gradhCorrection = gradhCorrection,
               XSPH = XSPH,
               correctVelocityGradient = correctVelocityGradient,
               densityUpdate = densityUpdate,
               HUpdate = HUpdate,
               epsTensile = epsTensile,
               nTensile = nTensile,
               damageRelieveRubble = damageRelieveRubble,
               negativePressureInDamage = negativePressureInDamage,
               strengthInDamage = strengthInDamage,
               xmin = xmin,
               xmax = xmax,
               ASPH = True)