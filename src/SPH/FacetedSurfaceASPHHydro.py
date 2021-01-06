from SolidSpheral3d import *

#-------------------------------------------------------------------------------
# A package which keeps the H tensors eigen vector associated with hmin aligned
# with the local surface normal.
#-------------------------------------------------------------------------------
# class FacetedSurfaceASPHSmoothingScale(Physics):

#     def __init__(self, surface):
#         self.surface = surface
#         Physics.__init__(self)
#         return

#     def evaluateDerivatives(self, t, dt, db, state, derivs):
#         return

#     def dt(self, db, state, derivs, t):
#         return pair_double_string(1e100, "No vote")

#     def registerState(self, db, state):
#         return

#     def registerDerivatives(self, db, derivs):
#         return

#     def label(self):
#         return "FacetedSurfaceASPHSmoothingScale"

#     def finalize(self, t, dt, db, state, derivs):
#         pos = state.vectorFields(HydroFieldNames.position)
#         H = state.symTensorFields(HydroFieldNames.H)
#         for nodeListi in xrange(pos.numFields):
#             for i in xrange(pos[nodeListi].numInternalNodes):
#                 havg2 = (3.0/H(nodeListi, i).Trace())**2
#                 norm = Vector()
#                 for facet self.surface.facets():
#                     r2 = (facet.position - pos(nodeListi, i)).magnitude2()
#                     wi = exp(-r2/havg2)
#                     norm += facet.normal * wi
#                 norm = norm.unitVector()
#                 T = rotationMatrix(norm)
#                 Ti = T.Transpose()
#                 if 

#-------------------------------------------------------------------------------
# A special version of the ASPH smoothing scale algorithm designed to keep the 
# hmin eigen vector tensor aligned with the local faceted surface normal.
#-------------------------------------------------------------------------------
class FacetedSurfaceASPHSmoothingScale(ASPHSmoothingScale):
    def __init__(self,
                 surface,
                 nodes2facets):
        self.surface = surface
        self.nodes2facets = nodes2facets
        self.poshash2facet = {}
        ASPHSmoothingScale.__init__(self)
        return

    def idealSmoothingScale(self,
                            H,
                            pos,
                            zerothMoment,
                            secondMoment,
                            W,
                            hmin,
                            hmax,
                            hminratio,
                            nPerh,
                            connectivityMap,
                            nodeListi,
                            i):

        H0 = ASPHSmoothingScale.idealSmoothingScale(self,
                                                    H,
                                                    pos,
                                                    zerothMoment,
                                                    secondMoment,
                                                    W,
                                                    hmin,
                                                    hmax,
                                                    hminratio,
                                                    nPerh,
                                                    connectivityMap,
                                                    nodeListi,
                                                    i)

        if nodeListi == self.imat and self.nodes2facets[i] >= 0:

            # Find the effective surface normal we want to use.
            fhat = Vector()
            verts = self.surface.vertices
            facets = self.surface.facets
            vertNorms = self.surface.vertexUnitNorms
            hashi = hash(tuple(pos))
            assert hashi in self.poshash2facet
            facet = facets[self.poshash2facet[hashi]]
            # fhat = facet.normal/max(1.0e-10, (pos - facet.position).magnitude2())
            # for i in facet.ipoints:
            #     fhat += vertNorms[i]/max(1.0e-10, (pos - verts[i]).magnitude2())
            # fhat = fhat.unitVector()
            fhat = facet.normal
            T1 = rotationMatrix(fhat)
            
            # Now decompose the trial H tensor and align hmin with the surface normal.
            Heigen = H0.eigenVectors()
            if ((Heigen.eigenValues.x > Heigen.eigenValues.y) and
                (Heigen.eigenValues.x > Heigen.eigenValues.z)):
                hvec = Heigen.eigenVectors.getColumn(0)
            elif ((Heigen.eigenValues.y > Heigen.eigenValues.x) and
                  (Heigen.eigenValues.y > Heigen.eigenValues.z)):
                hvec = Heigen.eigenVectors.getColumn(1)
            else:
                hvec = Heigen.eigenVectors.getColumn(2)
            T0 = rotationMatrix(hvec)
            H0.rotationalTransform(T0)
            H0.rotationalTransform(T1.Transpose())
        return H0

#-------------------------------------------------------------------------------
# The specialized hydro object using our very particular method of H adaptation.
#-------------------------------------------------------------------------------
class FacetedSurfaceASPHHydro(Physics):

    #---------------------------------------------------------------------------
    # Constructor.
    #---------------------------------------------------------------------------
    def __init__(self,
                 surface,
                 nodes2facets,
                 dataBase,
                 W,
                 WPi,
                 Q,
                 filter = 0.0,
                 cfl = 0.5,
                 useVelocityMagnitudeForDt = False,
                 compatibleEnergyEvolution = True,
                 evolveTotalEnergy = False,
                 gradhCorrection = False,
                 XSPH = True,
                 correctVelocityGradient = False,
                 sumMassDensityOverAllNodeLists = True,
                 densityUpdate = RigorousSumDensity,
                 HUpdate = IdealH,
                 epsTensile = 0.3,
                 nTensile = 4.0,
                 damageRelieveRubble = False,
                 negativePressureInDamage = False,
                 strengthInDamage = False,
                 xmin = Vector(-1e100, -1e100, -1e100),
                 xmax = Vector( 1e100,  1e100,  1e100)):
        Physics.__init__(self)
        self._smoothingScaleMethod = FacetedSurfaceASPHSmoothingScale(surface, nodes2facets)
        self._smoothingScaleMethod.imat = None
        if xmin is None:
            xmin = Vector.one
        self.hydro = SolidSPHHydroBase(self._smoothingScaleMethod,
                                       dataBase = dataBase,
                                       Q = Q,
                                       W = W,
                                       WPi = WPi,
                                       WGrad = W,
                                       filter = filter,
                                       cfl = cfl,
                                       useVelocityMagnitudeForDt = useVelocityMagnitudeForDt,
                                       compatibleEnergyEvolution = compatibleEnergyEvolution,
                                       evolveTotalEnergy = evolveTotalEnergy,
                                       gradhCorrection = gradhCorrection,
                                       XSPH = XSPH,
                                       correctVelocityGradient = correctVelocityGradient,
                                       sumMassDensityOverAllNodeLists = sumMassDensityOverAllNodeLists,
                                       densityUpdate = densityUpdate,
                                       HUpdate = HUpdate,
                                       epsTensile = epsTensile,
                                       nTensile = nTensile,
                                       damageRelieveRubble = damageRelieveRubble,
                                       negativePressureInDamage = negativePressureInDamage,
                                       strengthInDamage = strengthInDamage,
                                       xmin = xmin,
                                       xmax = xmax)
        return

    #---------------------------------------------------------------------------
    # dt
    #---------------------------------------------------------------------------
    def dt(self, db, state, derivs, t):
        return self.hydro.dt(db, state, derivs, t)

    #---------------------------------------------------------------------------
    # initializeProblemStartup
    #---------------------------------------------------------------------------
    def initializeProblemStartup(self, db):
        self.hydro.initializeProblemStartup(db)

    #---------------------------------------------------------------------------
    # registerState
    #---------------------------------------------------------------------------
    def registerState(self, db, state):
        self.hydro.registerState(db, state)

    #---------------------------------------------------------------------------
    # registerDerivatives
    #---------------------------------------------------------------------------
    def registerDerivatives(self, db, derivs):
        self.hydro.registerDerivatives(db, derivs)

    #---------------------------------------------------------------------------
    # evaluateDerivatives
    #---------------------------------------------------------------------------
    def evaluateDerivatives(self, t, dt, db, state, derivs):
        self.hydro.evaluateDerivatives(t, dt, db, state, derivs)

    #---------------------------------------------------------------------------
    # finalizeDerivatives
    #---------------------------------------------------------------------------
    def finalizeDerivatives(self, t, dt, db, state, derivs):
        self.hydro.finalizeDerivatives(t, dt, db, state, derivs)

    #---------------------------------------------------------------------------
    # finalize
    #---------------------------------------------------------------------------
    def finalize(self, t, dt, db, state, derivs):
        self.hydro.finalize(t, dt, db, state, derivs)

    #---------------------------------------------------------------------------
    # applyGhostBoundaries
    #---------------------------------------------------------------------------
    def applyGhostBoundaries(self, state, derivs):
        self.hydro.applyGhostBoundaries(state, derivs)

    #---------------------------------------------------------------------------
    # enforceBoundaries
    #---------------------------------------------------------------------------
    def enforceBoundaries(self, state, derivs):
        self.hydro.enforceBoundaries(state, derivs)

    #---------------------------------------------------------------------------
    # boundaryConditions
    #---------------------------------------------------------------------------
    def boundaryConditions(self):
        return self.hydro.boundaryConditions()

    #---------------------------------------------------------------------------
    # appendBoundary
    #---------------------------------------------------------------------------
    def appendBoundary(self, bc):
        self.hydro.appendBoundary(bc)

    #---------------------------------------------------------------------------
    # prependBoundary
    #---------------------------------------------------------------------------
    def prependBoundary(self, bc):
        self.hydro.prependBoundary(bc)

    #---------------------------------------------------------------------------
    # haveBoundary
    #---------------------------------------------------------------------------
    def haveBoundary(self, bc):
        return self.hydro.haveBoundary(bc)

    #---------------------------------------------------------------------------
    # clearBoundaries
    #---------------------------------------------------------------------------
    def clearBoundaries(self):
        self.hydro.clearBoundaries()

    #---------------------------------------------------------------------------
    # Override the pre-evaluateDerivative initialize step to update the 
    # node->facet mapping and build a lookup based on hashed position.  This
    # is necessary 'cause we don't pass the node index into the idealSmoothingScale 
    # algorithm above.
    #---------------------------------------------------------------------------
    def initialize(self, t, dt, dataBase, state, derivs):
        surface = self._smoothingScaleMethod.surface
        facets = surface.facets
        nodes2facets = self._smoothingScaleMethod.nodes2facets
        facets2facets = surface.facetFacetConnectivity
        posfl = state.vectorFields(HydroFieldNames.position)

        # Figure out which NodeList is the one we're pinning to the facets.
        if self._smoothingScaleMethod.imat is None:
            imat = 0
            while imat < len(posfl) and posfl[imat].nodeList().name != nodes2facets.nodeList().name:
                imat += 1
            assert imat < len(posfl)
            self._smoothingScaleMethod.imat = imat
        imat = self._smoothingScaleMethod.imat

        # First update the node->facet info.  We assume here the node can't have moved
        # too much since the last update.
        poshash2facet = {}
        #assert len(posfl) == 1
        pos = posfl[imat]
        n = pos.numInternalElements
        for i in xrange(n):
            if nodes2facets[i] >= 0:
                rmin = 1e300
                fimin = None
                for fi in facets2facets[nodes2facets[i]]:
                    r = facets[fi].distance(pos[i])
                    if r < rmin:
                        rmin = r
                        fimin = fi
                nodes2facets[i] = fimin
                poshash2facet[hash(tuple(pos[i]))] = fimin

        self._smoothingScaleMethod.poshash2facet = poshash2facet
        self.hydro.initialize(t, dt, dataBase, state, derivs)
        return

    # We need to restart some information in addition to the standard hydro variables.
    def label(self):
        return "FacetedSurfaceASPHHydro"

    def dumpState(self, file, path):
        file.write(self._smoothingScaleMethod.nodes2facets, path + "/nodes2facets")
        self.hydro.dumpState(file, path)
        return

    def restoreState(self, file, path):
        file.read(self._smoothingScaleMethod.nodes2facets, path + "/nodes2facets")
        self.hydro.restoreState(file, path)
        return

    #---------------------------------------------------------------------------
    # cfl
    #---------------------------------------------------------------------------
    @property
    def cfl(self):
        return self.hydro.cfl

    @cfl.setter
    def cfl(self, x):
        self.hydro.cfl = x

    #---------------------------------------------------------------------------
    # useVelocityMagnitudeForDt
    #---------------------------------------------------------------------------
    @property
    def useVelocityMagnitudeForDt(self):
        return self.hydro.useVelocityMagnitudeForDt

    @useVelocityMagnitudeForDt.setter
    def useVelocityMagnitudeForDt(self, x):
        self.hydro.useVelocityMagnitudeForDt = x

    #---------------------------------------------------------------------------
    # HEvolution
    #---------------------------------------------------------------------------
    @property
    def HEvolution(self):
        return self.hydro.HEvolution

    @HEvolution.setter
    def HEvolution(self, x):
        self.hydro.HEvolution = x

    #---------------------------------------------------------------------------
    # sumForMassDensity
    #---------------------------------------------------------------------------
    @property
    def sumForMassDensity(self):
        return self.hydro.sumForMassDensity

    @sumForMassDensity.setter
    def sumForMassDensity(self, x):
        self.hydro.sumForMassDensity = x

    #---------------------------------------------------------------------------
    # compatibleEnergyEvolution
    #---------------------------------------------------------------------------
    @property
    def compatibleEnergyEvolution(self):
        return self.hydro.compatibleEnergyEvolution

    @compatibleEnergyEvolution.setter
    def compatibleEnergyEvolution(self, x):
        self.hydro.compatibleEnergyEvolution = x

    #---------------------------------------------------------------------------
    # gradhCorrection
    #---------------------------------------------------------------------------
    @property
    def gradhCorrection(self):
        return self.hydro.gradhCorrection

    @gradhCorrection.setter
    def gradhCorrection(self, x):
        self.hydro.gradhCorrection = x

    #---------------------------------------------------------------------------
    # kernel
    #---------------------------------------------------------------------------
    def kernel(self):
        self.hydro.kernel()

    #---------------------------------------------------------------------------
    # PiKernel
    #---------------------------------------------------------------------------
    def PiKernel(self):
        self.hydro.PiKernel()

