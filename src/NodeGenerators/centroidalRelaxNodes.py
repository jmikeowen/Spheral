from math import *
import mpi

import Spheral
from SpheralVoronoiSiloDump import SpheralVoronoiSiloDump

#-------------------------------------------------------------------------------
# Centroidally (in mass) relax points allowing a linear density gradient.
#-------------------------------------------------------------------------------
def centroidalRelaxNodes(nodeListsAndBounds,
                         W,
                         rho,
                         gradrho = None,
                         boundaries = [],
                         maxIterations = 100,
                         fracTol = 1.0e-3,
                         correctionOrder = Spheral.LinearOrder,
                         centroidFrac = 0.5,
                         tessellationFileName = None):

    # Did we get passed a function or a constant for the density?
    if type(rho) == type(1.0):
        def rhofunc(posi):
            return rho
    else:
        rhofunc = rho
    rhofunc = rhofunc

    # What about the gradrho?  Did we get passed anything?
    if gradrho is None:
        gradrhofunc = None
    else:
        if type(gradrho) == type(1.0):
            def gradrhofunc(posi):
                return gradrho
        else:
            gradrhofunc = gradrho

    # Split out the NodeLists and bounding volumes (if available), depending on what was passed.
    if type(nodeListsAndBounds[0]) is tuple:
        nodeLists = [x[0] for x in nodeListsAndBounds]
        bounds = [x[1] for x in nodeListsAndBounds]
    else:
        nodeLists = nodeListsAndBounds
        bounds = []

    # Decide on our dimensionality and import the appropriate aliases.
    assert (isinstance(nodeLists[0], Spheral.NodeList1d) or
            isinstance(nodeLists[0], Spheral.NodeList2d) or
            isinstance(nodeLists[0], Spheral.NodeList3d))
    if isinstance(nodeLists[0], Spheral.NodeList1d):
        import Spheral1d as sph
        ndim = 1
    elif isinstance(nodeLists[0], Spheral.NodeList2d):
        import Spheral2d as sph
        ndim = 2
    else:
        import Spheral3d as sph
        ndim = 3

    # Build a local DataBase.
    db = sph.DataBase()
    for nodes in nodeLists:
        db.appendNodeList(nodes)

    # Get references to state in the NodeLists.
    pos = db.fluidPosition
    H = db.fluidHfield
    mass = db.fluidMass
    rho = db.fluidMassDensity

    # Prepare the storage for the point-wise fields.
    gradRho = db.newFluidVectorFieldList(sph.Vector.zero, "mass density gradient")
    surfacePoint = db.newFluidIntFieldList(0, "surface point")
    vol = db.newFluidScalarFieldList(0.0, "volume")
    deltaCentroid = db.newFluidVectorFieldList(sph.Vector.zero, "delta centroid")
    A = db.newFluidScalarFieldList(0.0, "A")
    B = db.newFluidVectorFieldList(sph.Vector.zero, "B")
    C = db.newFluidTensorFieldList(sph.Tensor.zero, "B")
    gradA = db.newFluidVectorFieldList(sph.Vector.zero, "gradA")
    gradB = db.newFluidTensorFieldList(sph.Tensor.zero, "gradB")
    gradC = db.newFluidThirdRankTensorFieldList(sph.ThirdRankTensor.zero, "gradC")
    m0 = db.newFluidScalarFieldList(0.0, "m0")
    m1 = db.newFluidVectorFieldList(sph.Vector.zero, "m1")
    m2 = db.newFluidSymTensorFieldList(sph.SymTensor.zero, "m2")
    m3 = db.newFluidThirdRankTensorFieldList(sph.ThirdRankTensor.zero, "m3")
    m4 = db.newFluidFourthRankTensorFieldList(sph.FourthRankTensor.zero, "m4")
    gradm0 = db.newFluidVectorFieldList(sph.Vector.zero, "gradm0")
    gradm1 = db.newFluidTensorFieldList(sph.Tensor.zero, "gradm1")
    gradm2 = db.newFluidThirdRankTensorFieldList(sph.ThirdRankTensor.zero, "gradm2")
    gradm3 = db.newFluidFourthRankTensorFieldList(sph.FourthRankTensor.zero, "gradm3")
    gradm4 = db.newFluidFifthRankTensorFieldList(sph.FifthRankTensor.zero, "gradm4")

    if tessellationFileName is None:
        cells = sph.FacetedVolumeFieldList()
    else:
        cells = db.newFluidFacetedVolumeFieldList(sph.FacetedVolume(), "cells")

    # Kick start the volume using m/rho.
    for nodeListi, nodes in enumerate(db.fluidNodeLists()):
        for i in xrange(nodes.numInternalNodes):
            vol[nodeListi][i] = mass(nodeListi, i)/rho(nodeListi, i)

    # We need the boundaries as a vector for calling iterateIdealH
    bound_vec = sph.vector_of_Boundary()
    for bc in boundaries:
        bound_vec.append(bc)

    # Same thing for the bounding volumes, if available
    boundingVolumes_vec = sph.vector_of_FacetedVolume()
    for b in bounds:
        boundingVolumes_vec.append(b)

    # Iterate until we converge or max out.
    iter = 0
    avgdelta = 2.0*fracTol
    while (iter < 2) or (iter < maxIterations and avgdelta > fracTol):
        iter += 1

        # Remove any old ghost nodes info, and update the mass density
        for nodeListi, nodes in enumerate(db.fluidNodeLists()):
            nodes.numGhostNodes = 0
            nodes.neighbor().updateNodes()

            for i in xrange(nodes.numInternalNodes):
                rho[nodeListi][i] = rhofunc(pos(nodeListi, i))

        # Create the new ghost nodes.
        for bc in boundaries:
            bc.setAllGhostNodes(db)
        for nodes in db.fluidNodeLists():
            nodes.neighbor().updateNodes()

        # Compute the new connectivity.
        db.updateConnectivityMap(False)
        cm = db.connectivityMap()

        # Compute the new volumes and centroids (note this uses the old rho gradient, not quite right,
        # but expedient/efficient).
        sph.computeVoronoiVolume(pos, H, rho, gradRho, cm, W.kernelExtent, boundingVolumes_vec, surfacePoint, vol, deltaCentroid, cells)
        
        # Apply boundary conditions.
        for bc in boundaries:
            bc.applyFieldListGhostBoundary(vol)
            bc.applyFieldListGhostBoundary(rho)
        for bc in boundaries:
            bc.finalizeGhostBoundary()

        # If the user provided a gradrho method, we can use it.  Otherwise we need to numerically evaluate
        # the density gradient.
        if gradrhofunc:
            for nodeListi, nodes in enumerate(db.fluidNodeLists()):
                for i in xrange(nodes.numInternalNodes):
                    gradRho[nodeListi][i] = gradrhofunc(pos(nodeListi, i))

        else:
            # Use RK to numerically compute the new mass density gradient.
            sph.computeCRKSPHMoments(cm, W, vol, pos, H, correctionOrder, sph.NodeCoupling(),
                                     m0, m1, m2, m3, m4, gradm0, gradm1, gradm2, gradm3, gradm4)
            sph.computeCRKSPHCorrections(m0, m1, m2, m3, m4, gradm0, gradm1, gradm2, gradm3, gradm4, H, correctionOrder,
                                         A, B, C, gradA, gradB, gradC)
            gradRho = sph.gradientCRKSPH(rho, pos, vol, H, A, B, C, gradA, gradB, gradC, cm, correctionOrder, W)
        
        # Displace the points and update point masses.
        avgdelta = 0.0
        for nodeListi, nodes in enumerate(db.fluidNodeLists()):
            for i in xrange(nodes.numInternalNodes):
                delta = centroidFrac * deltaCentroid(nodeListi, i)
                avgdelta += delta.magnitude()/vol(nodeListi, i)**(1.0/ndim)
                pos[nodeListi][i] += delta
                if bounds and not bounds[nodeListi].contains(pos[nodeListi][i]):
                    pos[nodeListi][i] = bounds[nodeListi].closestPoint(pos[nodeListi][i])
                    
                mass[nodeListi][i] = rho(nodeListi,i)*vol(nodeListi,i)
        avgdelta = mpi.allreduce(avgdelta, mpi.SUM)/mpi.allreduce(db.numInternalNodes, mpi.SUM)
        print "centroidalRelaxNodes iteration %i, avg delta frac %g" % (iter, avgdelta)

        # Update the H tensors a bit.
        sph.iterateIdealH(db, bound_vec, W, sph.ASPHSmoothingScale(), 2)

    # If requested, dump the final info to a diagnostic viz file.
    if tessellationFileName:
        dumper = SpheralVoronoiSiloDump(baseFileName = tessellationFileName,
                                        listOfFieldLists = [vol, surfacePoint, mass, deltaCentroid],
                                        boundaries = boundaries,
                                        cells = cells)
        dumper.dump(0.0, iter)

    return vol, surfacePoint
