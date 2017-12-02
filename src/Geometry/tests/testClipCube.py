#ATS:test(SELF, label="Polyhedron cube clipping tests")

import unittest
from math import *
import time
from SpheralTestUtilities import fuzzyEqual

from Spheral3d import *

# Create a global random number generator.
import random
rangen = random.Random()

#-------------------------------------------------------------------------------
# Test harness
#-------------------------------------------------------------------------------
class TestPolyhedronClipping(unittest.TestCase):

    #---------------------------------------------------------------------------
    # setUp
    #---------------------------------------------------------------------------
    def setUp(self):
        self.points = vector_of_Vector()
        for tup in [(0,0,0),
                    (1,0,0),
                    (0,1,0),
                    (1,1,0),
                    (0,0,1),
                    (1,0,1),
                    (0,1,1),
                    (1,1,1)]:
            self.points.append(Vector(*tup))
        self.cube = Polyhedron(self.points)
        self.ntests = 10000
        return

    #---------------------------------------------------------------------------
    # Clip with planes passing through the cube.
    #---------------------------------------------------------------------------
    def testClipInternalOnePlane(self):
        for i in xrange(self.ntests):
            planes1, planes2 = vector_of_Plane(), vector_of_Plane()
            p0 = Vector(rangen.uniform(0.0, 1.0),
                        rangen.uniform(0.0, 1.0),
                        rangen.uniform(0.0, 1.0))
            phat = Vector(rangen.uniform(-1.0, 1.0), 
                          rangen.uniform(-1.0, 1.0), 
                          rangen.uniform(-1.0, 1.0)).unitVector()
            planes1.append(Plane(p0,  phat))
            planes2.append(Plane(p0, -phat))
            chunk1 = Polyhedron(self.cube)
            chunk2 = Polyhedron(self.cube)
            clipFacetedVolumeByPlanes(chunk1, planes1)
            clipFacetedVolumeByPlanes(chunk2, planes2)
            # from PolyhedronFileUtilities import writePolyhedronOBJ
            # writePolyhedronOBJ(self.cube, "cube.obj")
            # writePolyhedronOBJ(chunk1, "chunk_ONE.obj")
            # writePolyhedronOBJ(chunk2, "chunk_TWO.obj")
            self.failUnless(fuzzyEqual(chunk1.volume + chunk2.volume, self.cube.volume),
                            "Plane clipping summing to wrong volumes: %s + %s != %s" % (chunk1.volume,
                                                                                        chunk2.volume,
                                                                                        self.cube.volume))
        return

    #---------------------------------------------------------------------------
    # Clip with planes passing through the cube.
    #---------------------------------------------------------------------------
    def testClipInternalTwoPlanes(self):
        for i in xrange(self.ntests):
            planes1 = vector_of_Plane()
            p0 = Vector(rangen.uniform(0.0, 1.0),
                        rangen.uniform(0.0, 1.0),
                        rangen.uniform(0.0, 1.0))
            for iplane in xrange(2):
                planes1.append(Plane(point = p0,
                                     normal = Vector(rangen.uniform(-1.0, 1.0), 
                                                     rangen.uniform(-1.0, 1.0), 
                                                     rangen.uniform(-1.0, 1.0)).unitVector()))
            planes2 = vector_of_Plane(planes1)
            planes3 = vector_of_Plane(planes1)
            planes4 = vector_of_Plane(planes1)
            planes2[0].normal = -planes2[0].normal
            planes3[1].normal = -planes3[1].normal
            planes4[0].normal = -planes4[0].normal
            planes4[1].normal = -planes4[1].normal
            chunk1 = Polyhedron(self.cube)
            chunk2 = Polyhedron(self.cube)
            chunk3 = Polyhedron(self.cube)
            chunk4 = Polyhedron(self.cube)
            clipFacetedVolumeByPlanes(chunk1, planes1)
            clipFacetedVolumeByPlanes(chunk2, planes2)
            clipFacetedVolumeByPlanes(chunk3, planes3)
            clipFacetedVolumeByPlanes(chunk4, planes4)
            # from PolyhedronFileUtilities import writePolyhedronOBJ
            # writePolyhedronOBJ(self.cube, "cube.obj")
            # writePolyhedronOBJ(chunk1, "chunk_1ONE_TWOPLANES.obj")
            # writePolyhedronOBJ(chunk2, "chunk_2TWO_TWOPLANES.obj")
            # writePolyhedronOBJ(chunk3, "chunk_3THREE_TWOPLANES.obj")
            # writePolyhedronOBJ(chunk4, "chunk_4FOUR_TWOPLANES.obj")
            self.failUnless(fuzzyEqual(chunk1.volume + chunk2.volume + chunk3.volume + chunk4.volume, self.cube.volume),
                            "Two plane clipping summing to wrong volumes: %s + %s + %s + %s = %s != %s" % (chunk1.volume,
                                                                                                           chunk2.volume,
                                                                                                           chunk3.volume,
                                                                                                           chunk4.volume,
                                                                                                           chunk1.volume + chunk2.volume + chunk3.volume + chunk4.volume,
                                                                                                           self.cube.volume))
        return

if __name__ == "__main__":
    # print "Waiting..."
    # raw_input()
    unittest.main()