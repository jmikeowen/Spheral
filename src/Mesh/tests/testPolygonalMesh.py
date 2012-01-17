#ATS:test(SELF,        label="PolygonalMesh serial unit tests")
#ATS:test(SELF, np=4,  label="PolygonalMesh serial (4 proc) tests")
#ATS:test(SELF, np=9,  label="PolygonalMesh serial (9 proc) tests")
#ATS:test(SELF, np=16, label="PolygonalMesh serial (16 proc) tests")

from math import *
import unittest
import time

from Spheral2d import *
from generateMesh import *
from SpheralTestUtilities import fuzzyEqual
from SpheralGnuPlotUtilities import *

#===============================================================================
# Load mpi, and figure out how may domains to set up, and which domain we are.
#===============================================================================
import mpi
rank = mpi.rank
numDomains = mpi.procs
nxproc = int(sqrt(numDomains))
assert nxproc*nxproc == numDomains

#===============================================================================
# Create a global random number generator.
#===============================================================================
import random
rangen = random.Random()

#===============================================================================
# Some boundary conditions.
#===============================================================================
x0, x1 = 0.0, 1.0
y0, y1 = 0.0, 1.0

nx, ny = 16*nxproc, 16*nxproc
n = nx*ny
nperdomain = n / numDomains
nxcell = KeyTraits.maxKey1d/4
nycell = nxcell
assert nx < nxcell
ncell = nxcell*nycell
dxcell = (x1 - x0)/nxcell
dycell = (y1 - y0)/nycell

xmin = Vector(x0, y0)
xmax = Vector(x1, y1)

xbc0 = ReflectingBoundary(Plane(Vector(x0, y0), Vector(1.0, 0.0)))
ybc0 = ReflectingBoundary(Plane(Vector(x0, y0), Vector(0.0, 1.0)))
bclist = [xbc0, ybc0]
if numDomains > 1:
    bclist.append(BoundingVolumeDistributedBoundary.instance())
bclist = []

#===============================================================================
# Iterate the H's to something reasonable.
#===============================================================================
def iterateThoseHs(nodes):
    db = DataBase()
    db.appendNodeList(nodes)
    for bc in bclist:
        bc.setAllGhostNodes(db)
        nodes.neighbor().updateNodes()
    for bc in bclist:
        bc.finalizeGhostBoundary()
    nodes.neighbor().updateNodes()
    vecbound = vector_of_Boundary()
    for bc in bclist:
        vecbound.append(bc)
    WT = TableKernel(BSplineKernel(), 1000)
    smooth = SPHSmoothingScale()
    iterateIdealH(db, vecbound, WT, smooth,
                  tolerance = 1.0e-4)
    return

from SpheralGnuPlotUtilities import *
p = None

#===============================================================================
# A counter to help in creating unique NodeList names.
#===============================================================================
itest = 0

#===============================================================================
# Test class for tests to apply to all meshes.
#===============================================================================
class PolygonalMeshGenericTests:

    #---------------------------------------------------------------------------
    # Test numbers of elements.
    #---------------------------------------------------------------------------
    def testPolygonalMeshNums0(self):
        t0 = time.clock()
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax,
                                           generateVoid = False,
                                           generateParallelConnectivity = False)
        print "Required %f seconds to generate mesh" % (time.clock() - t0)
        assert mesh.numZones == self.nodes.numInternalNodes
##         p = plotPolygonalMesh(mesh, persist=True)
##         p("set xrange [-0.1:1.1]; set yrange [-0.1:1.1]; set size square"); p.refresh()
##         d = Gnuplot.Data([self.pos[i].x for i in xrange(self.nodes.numInternalNodes)],
##                          [self.pos[i].y for i in xrange(self.nodes.numInternalNodes)],
##                          with_ = "points"
##                          )
##         p.replot(d)

    #---------------------------------------------------------------------------
    # Test element IDs.
    #---------------------------------------------------------------------------
    def testPolygonalMeshElementIDs(self):
        t0 = time.clock()
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax,
                                           generateVoid = False,
                                           generateParallelConnectivity = False)
        print "Required %f seconds to generate mesh" % (time.clock() - t0)
        for i in xrange(self.nodes.numInternalNodes):
            node = mesh.node(i)
            assert node.ID == i
        for i in xrange(mesh.numEdges):
            edge = mesh.edge(i)
            assert edge.ID == i
        for i in xrange(mesh.numFaces):
            face = mesh.face(i)
            assert face.ID == i
        for i in xrange(mesh.numZones):
            zone = mesh.zone(i)
            assert zone.ID == i
        return

    #---------------------------------------------------------------------------
    # Test that the zones in the mesh correspond to the correct seed nodes.
    #---------------------------------------------------------------------------
    def testPolygonalMeshZoneOrder(self):
        t0 = time.clock()
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax,
                                           generateVoid = False,
                                           generateParallelConnectivity = False)
        print "Required %f seconds to generate mesh" % (time.clock() - t0)
        assert mesh.numZones >= self.nodes.numInternalNodes
        for i in xrange(self.nodes.numInternalNodes):
            zonehull = mesh.zone(i).convexHull()
            assert zonehull.contains(self.pos[i])
        return

    #---------------------------------------------------------------------------
    # Test the minimum scale.
    #---------------------------------------------------------------------------
    def testPolygonalMeshMinimumScale(self):
        t0 = time.clock()
        mesh, void = generatePolygonalMesh([self.nodes],
                                      xmin = xmin,
                                      xmax = xmax)
        print "Required %f seconds to generate mesh" % (time.clock() - t0)
        self.failUnless(mesh.minimumScale <= self.dxmin, 
                        "Scales don't match:  %g %g" % (mesh.minimumScale, self.dxmin))
        return

    #---------------------------------------------------------------------------
    # Test the parallel domain info.
    #---------------------------------------------------------------------------
    def testPolygonalMeshParallel(self):
        t0 = time.clock()
        mesh, void = generatePolygonalMesh([self.nodes],
                                      xmin = xmin,
                                      xmax = xmax,
                                      generateParallelConnectivity = True)
        print "Required %f seconds to generate mesh" % (time.clock() - t0)

##         p = plotPolygonalMesh(mesh, persist=True)
##         p("set xrange [-0.1:1.1]; set yrange [-0.1:1.1]; set size square"); p.refresh()
##         d = Gnuplot.Data([self.pos[i].x for i in xrange(self.nodes.numInternalNodes)],
##                          [self.pos[i].y for i in xrange(self.nodes.numInternalNodes)],
##                          with_ = "points"
##                          )
##         p.replot(d)

        neighborDomains = [int(x) for x in mesh.neighborDomains]
        sharedNodes = []
        for ll in mesh.sharedNodes:
            sharedNodes.append([int(x) for x in ll])
        assert len(neighborDomains) == len(mesh.sharedNodes)

##         # Check the correct domains are talking to each other.
##         nxproc = int(sqrt(numDomains))
##         assert nxproc*nxproc == numDomains
##         ixproc = rank % nxproc
##         iyproc = rank / nxproc
##         neighborDomainsAnswer = []
##         for iy in xrange(max(0, iyproc - 1), min(nxproc, iyproc + 2)):
##             for ix in xrange(max(0, ixproc - 1), min(nxproc, ixproc + 2)):
##                 if not (ix == ixproc and iy == iyproc):
##                     neighborDomainsAnswer.append(ix + iy*nxproc)
##         ok = mpi.allreduce((neighborDomains == neighborDomainsAnswer), mpi.MIN)
##         self.failUnless(ok, "Strange neighbor domains for %i : %s ?= %s" % (rank, neighborDomains, neighborDomainsAnswer))

        # Check that the communicated mesh nodes are consistent.
        boxInv = xmax - xmin
        boxInv = Vector(1.0/boxInv.x, 1.0/boxInv.y)
        for sendProc in xrange(numDomains):
            numChecks = mpi.bcast(len(neighborDomains), root=sendProc)
            assert mpi.allreduce(numChecks, mpi.MIN) == mpi.allreduce(numChecks, mpi.MAX)
            for k in xrange(numChecks):
                if rank == sendProc:
                    ksafe = k
                else:
                    ksafe = 0
                recvProc = mpi.bcast(neighborDomains[ksafe], root=sendProc)
                recvHashes = mpi.bcast([hashPosition(mesh.node(i).position(), xmin, xmax, boxInv) for i in sharedNodes[ksafe]], root=sendProc)
                recvPos = mpi.bcast([str(mesh.node(i).position()) for i in sharedNodes[ksafe]], root=sendProc)
                ok = True
                msg = ""
                if rank == recvProc:
                    assert sendProc in neighborDomains
                    kk = neighborDomains.index(sendProc)
                    assert kk < len(sharedNodes)
                    ok = ([hashPosition(mesh.node(i).position(), xmin, xmax, boxInv) for i in sharedNodes[kk]] == recvHashes)
                    msg = ("Shared node indicies don't match %i %i\n   %s != %s\n    %s\n    %s" %
                           (rank, sendProc, 
                            str([hashPosition(mesh.node(i).position(), xmin, xmax, boxInv) for i in sharedNodes[kk]]),
                            recvHashes,
                            [str(mesh.node(i).position()) for i in sharedNodes[kk]],
                            recvPos))
                self.failUnless(mpi.allreduce(ok, mpi.MIN), msg)

        # Check that all shared nodes have been found.
        myHashes = [hashPosition(mesh.node(i).position(), xmin, xmax, boxInv) for i in xrange(mesh.numNodes)]
        myHashSet = set(myHashes)
        for sendProc in xrange(numDomains):
            theirHashSet = mpi.bcast(myHashSet, root=sendProc)
            if sendProc != mpi.rank:
                commonHashes = myHashSet.intersection(theirHashSet)
                self.failIf(len(commonHashes) > 0 and (not sendProc in neighborDomains),
                            "Missed a neighbor domain : %i %i : %i" % (mpi.rank, sendProc, len(commonHashes)))
                self.failIf(len(commonHashes) == 0 and (sendProc in neighborDomains),
                            "Erroneously communicating between domains : %i %i" % (mpi.rank, sendProc))
                if len(commonHashes) > 0:
                    k = neighborDomains.index(sendProc)
                    self.failUnless(len(commonHashes) == len(sharedNodes[k]),
                                    "Size of shared nodes does not match: %i %i : %i %i" % (mpi.rank, sendProc,
                                                                                            len(commonHashes),
                                                                                            len(sharedNodes[k])))
                    sharedHashes = set([myHashes[i] for i in sharedNodes[k]])
                    self.failUnless(sharedHashes == commonHashes, "Set of common hashes does not match")

        return

    #---------------------------------------------------------------------------
    # Test the mesh coordinates hash uniquely.
    #---------------------------------------------------------------------------
    def testPolygonalMeshHash(self):
        t0 = time.clock()
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax)
        print "Required %f seconds to generate mesh" % (time.clock() - t0)
        pos = [mesh.zone(i).position() for i in xrange(mesh.numZones)] + [mesh.node(i).position() for i in xrange(mesh.numNodes)]
        boxInv = xmax - xmin
        boxInv = Vector(1.0/boxInv.x, 1.0/boxInv.y)
        hashes = [hashPosition(x, xmin, xmax, boxInv) for x in pos]
        blarg = zip(hashes, pos)
        blarg.sort()
        for i in xrange(len(blarg) - 1):
            hash0 = blarg[i][0]
            hash1 = blarg[i+1][0]
            self.failIf(hash0 == hash1,
                        "%i: Non-unique hash:  %i %i %s %s %s %s %g" % (mpi.rank, i, mesh.numZones, str(hash0), str(hash1), str(blarg[i][1]), str(blarg[i+1][1]), (blarg[i][1] - blarg[i+1][1]).magnitude()))
        return

    #---------------------------------------------------------------------------
    # Test the zones of the nodes.
    #---------------------------------------------------------------------------
    def testPolygonalMeshNodeZones(self):
        t0 = time.clock()
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax)
        print "Required %f seconds to generate mesh" % (time.clock() - t0)
        answer = {}
        for inode in xrange(mesh.numNodes):
            answer[inode] = set()
        for izone in xrange(mesh.numZones):
            nodeIDs = mesh.zone(izone).nodeIDs
            for inode in nodeIDs:
                answer[inode].add(izone)
        for inode in xrange(mesh.numNodes):
            zoneIDs = mesh.node(inode).zoneIDs
            for izone in zoneIDs:
                self.failUnless(izone in answer[inode] or izone == PolygonalMesh.UNSETID,
                                "Missing zone %i for set in node %i: %s %s" %
                                (izone, inode, [x for x in zoneIDs], answer[inode]))

    #---------------------------------------------------------------------------
    # Test consistency of zone adjacency via node connection.
    #---------------------------------------------------------------------------
    def testPolygonalZoneAdjacency(self):
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax)
        for izone in xrange(mesh.numZones):
            nodeIDs = mesh.zone(izone).nodeIDs
            for inode in nodeIDs:
                self.failUnless(izone in mesh.node(inode).zoneIDs,
                                "Missing zone %i in neighbors for node %i : %s" % (izone, inode, list(mesh.node(inode).zoneIDs)))

    #---------------------------------------------------------------------------
    # Test the opposite zones across faces.
    #---------------------------------------------------------------------------
    def testPolygonalMeshOppZones(self):
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax)
        answer = [[] for i in xrange(mesh.numFaces)]
        for izone in xrange(mesh.numZones):
            faces = mesh.zone(izone).faceIDs
            for iface in faces:
                answer[iface].append(izone)

        for iface in xrange(mesh.numFaces):
            face = mesh.face(iface)
            zoneIDs = answer[iface]
            assert len(zoneIDs) in (1, 2)
            if len(zoneIDs) == 2:
                self.failUnless(face.oppositeZoneID(zoneIDs[0]) == zoneIDs[1],
                                "Bad opposites:  (%i, %i) != (%i, %i)" %
                                (zoneIDs[0], zoneIDs[1],
                                 face.oppositeZoneID(zoneIDs[0]), face.oppositeZoneID(zoneIDs[1])))
                self.failUnless(face.oppositeZoneID(zoneIDs[1]) == zoneIDs[0],
                                "Bad opposites:  (%i, %i) != (%i, %i)" %
                                (zoneIDs[0], zoneIDs[1],
                                 face.oppositeZoneID(zoneIDs[0]), face.oppositeZoneID(zoneIDs[1])))
            else:
                assert face.oppositeZoneID(zoneIDs[0]) == PolygonalMesh.UNSETID

    #---------------------------------------------------------------------------
    # Test the global mesh node IDs.
    #---------------------------------------------------------------------------
    def testGlobalMeshNodeIDs(self):
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax,
                                           generateParallelConnectivity = True)
        globalIDs = mesh.globalMeshNodeIDs()

        # Check that all our local IDs are unique.
        uniqueIDs = set()
        for i in globalIDs:
            uniqueIDs.add(i)
        self.failUnless(len(uniqueIDs) == len(globalIDs),
                        "Global mesh node IDs not unique!  %i != %i" % (len(globalIDs), len(uniqueIDs)))

        # Check that the IDs are unique and consistent across domains.
        if mpi.procs > 1:
            neighbors = mesh.neighborDomains
            sharedNodes = mesh.sharedNodes
            assert len(neighbors) == len(sharedNodes)

            # Translate to the shared nodes to global IDs.
            sharedGlobalIDs = [[globalIDs[i] for i in localIDs] for localIDs in sharedNodes]
            assert len(sharedGlobalIDs) == len(neighbors)

            # Do non-blocking sends to all our neighbors.
            sendRequests = []
            for neighbor, ids in zip(neighbors, sharedGlobalIDs):
                sendRequests.append(mpi.isend(ids, dest=neighbor))
            assert len(sendRequests) == len(neighbors)

            # Recv the IDs from our neighbors and do the testing.
            for neighbor, localIDs in zip(neighbors, sharedGlobalIDs):
                otherIDs = mpi.recv(source=neighbor)[0]
                self.failUnless(otherIDs == list(localIDs),
                                "Global IDs don't match between domains %i <-> %i\n%s\n%s" % (mpi.rank, neighbor, list(localIDs), otherIDs))

            # Wait until all our sends have completed.
            for req in sendRequests:
                req.Wait()

    #---------------------------------------------------------------------------
    # Test the bounding surface.
    #---------------------------------------------------------------------------
    def testZBoundingSurface(self):
        mesh, void = generatePolygonalMesh([self.nodes],
                                           xmin = xmin,
                                           xmax = xmax,
                                           generateVoid = False,
                                           generateParallelConnectivity = True)
        bs = mesh.boundingSurface()

##         if mpi.rank == 0:
##             p = plotPolygon(bs, plotNormals=True, persist=True)
##             p("set xrange [-0.1:1.1]; set yrange [-0.1:1.1]; set size square"); p.refresh()

##         for facet in bs.facets():
##             self.failUnless(facet.compare(Vector(0.5, 0.5)) == -1,
##                             "Bluh? %s %s %i" % (facet.position, facet.normal, 
##                                                 facet.compare(Vector(0.5, 0.5))))

        # Check that all the generators are contained.
        pos = self.nodes.positions()
        for i in xrange(self.nodes.numInternalNodes):
            self.failUnless(bs.contains(pos[i]),
                            "Failed containment for generator %i @ %s" % (i, pos[i]))

        # Check that all mesh nodes are contained.
        for i in xrange(mesh.numNodes):
            self.failUnless(bs.contains(mesh.node(i).position()),
                            "Failed containment for mesh node %i @ %s" % (i, mesh.node(i).position()))

        return

#===============================================================================
# Create a uniformly spaced nodes/mesh.
#===============================================================================
class UniformPolygonalMeshTests(unittest.TestCase, PolygonalMeshGenericTests):

    #---------------------------------------------------------------------------
    # Create the NodeList we'll use for generating the mesh.
    #---------------------------------------------------------------------------
    def setUp(self):
        global itest
        eos = GammaLawGasMKS(5.0/3.0, 1.0)
        self.nodes = makeFluidNodeList("test nodes %i" % itest, eos,
                                       numInternal = nperdomain,
                                       nPerh = 2.01,
                                       hmin = 1e-5,
                                       hmax = 0.3)
        itest += 1
        self.pos = self.nodes.positions()
        self.H = self.nodes.Hfield()

        # Generate positions and split them up between domains appropriately.
        dxproc = (x1 - x0)/nxproc
        dyproc = (y1 - y0)/nxproc
        ixproc = rank % nxproc
        iyproc = rank / nxproc
        xminproc = Vector(x0 + ixproc*dxproc, y0 + iyproc*dyproc)
        xmaxproc = Vector(x0 + (ixproc + 1)*dxproc, y0 + (iyproc + 1)*dyproc)

        dxavg = (x1 - x0)/nx
        dyavg = (y1 - y0)/ny
        self.dxmin = dxavg
        xynodes_all = [Vector(x0 + (i % nx + 0.5)*dxavg, y0 + (i / nx + 0.5)*dyavg) for i in xrange(n)]
        xynodes = [v for v in xynodes_all if testPointInBox(v, xminproc, xmaxproc)]
        assert len(xynodes) == nperdomain
        assert mpi.allreduce(len(xynodes), mpi.SUM) == n

        # We now have the positions for each domain appropriately divided, so shuffle
        # the local positions.
        random.shuffle(xynodes)

        # Now we can set the node conditions.
        for i in xrange(nperdomain):
            self.pos[i] = xynodes[i]
            self.H[i] = SymTensor(1.0/(2.0*dxavg), 0.0,
                                  0.0, 1.0/(2.0*dyavg))
        self.nodes.neighbor().updateNodes()

        # Fix up the H's.
        #iterateThoseHs(self.nodes)
        return

    #---------------------------------------------------------------------------
    # Standard destructor.
    #---------------------------------------------------------------------------
    def tearDown(self):
        del self.nodes
        return

#===============================================================================
# Create randomly spaced set of nodes in the unit square.
#===============================================================================
class RandomPolygonalMeshTests(unittest.TestCase, PolygonalMeshGenericTests):

    #---------------------------------------------------------------------------
    # Create the NodeList we'll use for generating the mesh.
    #---------------------------------------------------------------------------
    def setUp(self):
        global itest
        eos = GammaLawGasMKS(5.0/3.0, 1.0)
        self.nodes = makeFluidNodeList("test nodes %i" % itest, eos,
                                       numInternal = nperdomain,
                                       nPerh = 2.01,
                                       hmin = 1.0e-5,
                                       hmax = 0.3)
        itest += 1
        self.pos = self.nodes.positions()
        self.H = self.nodes.Hfield()

        # Figure out the domain bounding volumes.
        dxproc = (x1 - x0)/nxproc
        dyproc = (y1 - y0)/nxproc
        ixproc = rank % nxproc
        iyproc = rank / nxproc
        xminproc = Vector(x0 + ixproc*dxproc, y0 + iyproc*dyproc)
        xmaxproc = Vector(x0 + (ixproc + 1)*dxproc, y0 + (iyproc + 1)*dyproc)

        # Randomly seed the generators.  We choose from random cells in order
        # to keep nodes from getting too close together.
        xynodes_all = []
        occupiedCells = set()
        for k in xrange(n):
            i = rangen.randint(0, ncell)
            while i in occupiedCells:
                i = rangen.randint(0, ncell)
            ix = i % nxcell
            iy = i / nxcell
            xynodes_all.append(Vector((ix + 0.5)*dxcell, (iy + 0.5)*dycell))
            occupiedCells.add(i)
        assert len(occupiedCells) == n
        xynodes_all = mpi.bcast(xynodes_all)
        xynodes = [v for v in xynodes_all if testPointInBox(v, xminproc, xmaxproc)]
        dxavg = (x1 - x0)/nx
        dyavg = (y1 - y0)/ny
        self.dxmin = dxavg
        assert mpi.allreduce(len(xynodes), mpi.SUM) == n

        # Now we can set the node conditions.
        self.nodes.numInternalNodes = len(xynodes)
        for i in xrange(len(xynodes)):
            self.pos[i] = xynodes[i]
            self.H[i] = SymTensor(1.0/(2.0*dxavg), 0.0,
                                  0.0, 1.0/(2.0*dyavg))
        self.nodes.neighbor().updateNodes()

        # Fix up the H's.
        iterateThoseHs(self.nodes)
        return

    #---------------------------------------------------------------------------
    # Standard destructor.
    #---------------------------------------------------------------------------
    def tearDown(self):
        del self.nodes
        return

#===============================================================================
# Run the tests
#===============================================================================
if __name__ == "__main__":
    unittest.main()
