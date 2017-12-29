#ATS:test(SELF, label="PolyClipper 2D (polygon) tests")

import unittest
from math import *
import time
from PolyhedronFileUtilities import *

from Spheral2d import *
from SpheralTestUtilities import fuzzyEqual

# Create a global random number generator.
import random
rangen = random.Random()

# Make a square
points = vector_of_Vector()
for coords in [(0,0), (1,0), (0,1), (1,1)]:
    points.append(Vector(*coords))
square = Polygon(points)

# Make a non-convex notched thingy.
points = vector_of_Vector()
for coords in [(0,0), (4,0), (4,2), (3,2), (2,1), (1,2), (0,2)]:
    points.append(Vector(*coords))
facets = vector_of_vector_of_unsigned(7, vector_of_unsigned(2))
for i in xrange(7):
    facets[i][0] = i
    facets[i][1] = (i + 1) % 7
notchedthing = Polygon(points, facets)

#-------------------------------------------------------------------------------
# Test harness
#-------------------------------------------------------------------------------
class TestPolyClipper2d(unittest.TestCase):

    #---------------------------------------------------------------------------
    # setUp
    #---------------------------------------------------------------------------
    def setUp(self):
        self.polygons = [square, notchedthing]
        self.ntests = 1000
        return

    #---------------------------------------------------------------------------
    # Spheral::Polygon --> PolyClipper::Polygon
    #---------------------------------------------------------------------------
    def testConvertToPolygon(self):
        for poly in self.polygons:
            PCpoly = PolyClipper.Polygon()
            PolyClipper.convertToPolygon(PCpoly, poly)
            assert poly.vertices().size() == PCpoly.size()
            vol, centroid = PolyClipper.moments(PCpoly)
            self.failUnless(vol == poly.volume,
                            "Volume comparison failure: %g != %g" % (vol, poly.volume))
            self.failUnless(centroid == poly.centroid(),
                            "Centroid comparison failure: %s != %s" % (centroid, poly.centroid))


    #---------------------------------------------------------------------------
    # PolyClipper::Polygon --> Spheral::Polygon
    #---------------------------------------------------------------------------
    def testConvertFromPolygon(self):
        for poly0 in self.polygons:
            PCpoly = PolyClipper.Polygon()
            PolyClipper.convertToPolygon(PCpoly, poly0)
            assert poly0.vertices().size() == PCpoly.size()
            poly1 = Polygon()
            PolyClipper.convertFromPolygon(poly1, PCpoly)
            assert poly1 == poly0

    #---------------------------------------------------------------------------
    # PolyClipper::copyPolygon
    #---------------------------------------------------------------------------
    def testCopyPolygon(self):
        for poly0 in self.polygons:
            PCpoly0 = PolyClipper.Polygon()
            PolyClipper.convertToPolygon(PCpoly0, poly0)
            assert poly0.vertices().size() == PCpoly0.size()
            PCpoly1 = PolyClipper.Polygon()
            PolyClipper.copyPolygon(PCpoly1, PCpoly0)
            assert PCpoly0.size() == PCpoly1.size()
            vol0, centroid0 = PolyClipper.moments(PCpoly0)
            vol1, centroid1 = PolyClipper.moments(PCpoly1)
            assert vol0 == vol1
            assert centroid0 == centroid1

    #---------------------------------------------------------------------------
    # Clip with planes passing through the polygon.
    #---------------------------------------------------------------------------
    def testClipInternalOnePlane(self):
        for poly in self.polygons:
            PCpoly = PolyClipper.Polygon()
            PolyClipper.convertToPolygon(PCpoly, poly)
            for i in xrange(self.ntests):
                planes1, planes2 = vector_of_Plane(), vector_of_Plane()
                p0 = Vector(rangen.uniform(0.0, 1.0),
                            rangen.uniform(0.0, 1.0))
                phat = Vector(rangen.uniform(-1.0, 1.0), 
                              rangen.uniform(-1.0, 1.0)).unitVector()
                planes1.append(Plane(p0,  phat))
                planes2.append(Plane(p0, -phat))
                PCchunk1 = PolyClipper.Polygon()
                PCchunk2 = PolyClipper.Polygon()
                PolyClipper.copyPolygon(PCchunk1, PCpoly)
                PolyClipper.copyPolygon(PCchunk2, PCpoly)
                PolyClipper.clipPolygon(PCchunk1, planes1)
                PolyClipper.clipPolygon(PCchunk2, planes2)
                chunk1 = Polygon()
                chunk2 = Polygon()
                PolyClipper.convertFromPolygon(chunk1, PCchunk1)
                PolyClipper.convertFromPolygon(chunk2, PCchunk2)
                success = fuzzyEqual(chunk1.volume + chunk2.volume, poly.volume)
                # if not success:
                #     print "Poly:\n", poly
                #     print "Chunk 1:\n ", chunk1
                #     print "Chunk 2:\n ", chunk2
                #     writePolyhedronOBJ(poly, "poly.obj")
                #     writePolyhedronOBJ(chunk1, "chunk_ONE.obj")
                #     writePolyhedronOBJ(chunk2, "chunk_TWO.obj")
                self.failUnless(success,
                                "Plane clipping summing to wrong volumes: %s + %s != %s" % (chunk1.volume,
                                                                                            chunk2.volume,
                                                                                            poly.volume))
        return

    # #---------------------------------------------------------------------------
    # # Clip with planes passing outside the polygon -- null test.
    # #---------------------------------------------------------------------------
    # def testNullClipOnePlane(self):
    #     for poly in self.polygons:
    #         for i in xrange(self.ntests):
    #             r = rangen.uniform(2.0, 100.0) * (poly.xmax - poly.xmin).magnitude()
    #             theta = rangen.uniform(0.0, 2.0*pi)
    #             phat = Vector(cos(theta), sin(theta))
    #             p0 = poly.centroid() + r*phat
    #             planes = vector_of_Plane()
    #             planes.append(Plane(p0, -phat))
    #             chunk = Polygon(poly)
    #             clipFacetedVolumeByPlanes(chunk, planes)
    #             success = (chunk.volume == poly.volume)
    #             if not success:
    #                 writePolyhedronOBJ(poly, "poly.obj")
    #                 writePolyhedronOBJ(chunk, "chunk.obj")
    #             self.failUnless(success,
    #                             "Null plane clipping failure: %s != %s" % (chunk.volume, poly.volume))
    #     return

    # #---------------------------------------------------------------------------
    # # Clip with planes passing outside the polygon and rejecting the whole thing.
    # #---------------------------------------------------------------------------
    # def testFullClipOnePlane(self):
    #     for poly in self.polygons:
    #         for i in xrange(self.ntests):
    #             planes = vector_of_Plane()
    #             r = rangen.uniform(2.0, 100.0) * (poly.xmax - poly.xmin).magnitude()
    #             theta = rangen.uniform(0.0, 2.0*pi)
    #             phat = Vector(cos(theta), sin(theta))
    #             p0 = poly.centroid() + r*phat
    #             planes.append(Plane(p0, phat))
    #             chunk = Polygon(poly)
    #             clipFacetedVolumeByPlanes(chunk, planes)
    #             success = (chunk.volume == 0.0)
    #             if not success:
    #                 writePolyhedronOBJ(poly, "poly.obj")
    #                 writePolyhedronOBJ(chunk, "chunk.obj")
    #             self.failUnless(success,
    #                             "Full plane clipping failure: %s != %s" % (chunk.volume, poly.volume))
    #     return

    # #---------------------------------------------------------------------------
    # # Clip with planes passing through the polygon.
    # #---------------------------------------------------------------------------
    # def testClipInternalTwoPlanes(self):
    #     for poly in self.polygons:
    #         for i in xrange(self.ntests):
    #             planes1 = vector_of_Plane()
    #             p0 = Vector(rangen.uniform(0.0, 1.0),
    #                         rangen.uniform(0.0, 1.0))
    #             for iplane in xrange(2):
    #                 planes1.append(Plane(point = p0,
    #                                      normal = Vector(rangen.uniform(-1.0, 1.0), 
    #                                                      rangen.uniform(-1.0, 1.0)).unitVector()))
    #             planes2 = vector_of_Plane(planes1)
    #             planes3 = vector_of_Plane(planes1)
    #             planes4 = vector_of_Plane(planes1)
    #             planes2[0].normal = -planes2[0].normal
    #             planes3[1].normal = -planes3[1].normal
    #             planes4[0].normal = -planes4[0].normal
    #             planes4[1].normal = -planes4[1].normal
    #             chunk1 = Polygon(poly)
    #             chunk2 = Polygon(poly)
    #             chunk3 = Polygon(poly)
    #             chunk4 = Polygon(poly)
    #             clipFacetedVolumeByPlanes(chunk1, planes1)
    #             clipFacetedVolumeByPlanes(chunk2, planes2)
    #             clipFacetedVolumeByPlanes(chunk3, planes3)
    #             clipFacetedVolumeByPlanes(chunk4, planes4)
    #             success = fuzzyEqual(chunk1.volume + chunk2.volume + chunk3.volume + chunk4.volume, poly.volume)
    #             if not success:
    #                 writePolyhedronOBJ(poly, "poly.obj")
    #                 writePolyhedronOBJ(chunk1, "chunk_1ONE_TWOPLANES.obj")
    #                 writePolyhedronOBJ(chunk2, "chunk_2TWO_TWOPLANES.obj")
    #                 writePolyhedronOBJ(chunk3, "chunk_3THREE_TWOPLANES.obj")
    #                 writePolyhedronOBJ(chunk4, "chunk_4FOUR_TWOPLANES.obj")
    #             self.failUnless(success,
    #                             "Two plane clipping summing to wrong volumes: %s + %s + %s + %s = %s != %s" % (chunk1.volume,
    #                                                                                                            chunk2.volume,
    #                                                                                                            chunk3.volume,
    #                                                                                                            chunk4.volume,
    #                                                                                                            chunk1.volume + chunk2.volume + chunk3.volume + chunk4.volume,
    #                                                                                                            poly.volume))
    #     return

if __name__ == "__main__":
    unittest.main()