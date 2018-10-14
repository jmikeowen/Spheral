#-------------------------------------------------------------------------------
# Polygon
#-------------------------------------------------------------------------------
from PYB11Generator import *

@PYB11cppname("GeomPolygon")
class Polygon:

    typedefs = """
    typedef GeomPolygon Polygon;
    typedef GeomPolygon::Vector Vector;
    typedef GeomPolygon::Facet Facet;
"""

    #...........................................................................
    # Constructors
    def pyinit(self):
        "Default constructor"

    def pyinit1(self,
                points = "const std::vector<Vector>&"):
        """Note this constructor constructs the convex hull of the given points,
meaning that the full set of points passed in may not appear in the vertices."""

    def pyinit2(self,
                points = "const std::vector<Vector>&",
                facetIndices = "const std::vector<std::vector<unsigned> >&"):
        "Construct with explicit vertices and facets"

    #...........................................................................
    # Methods
    @PYB11const
    def contains(self,
                 point = "const Vector&",
                 countBoundary = ("const bool", "true"),
                 tol = ("const double", "1.0e-8")):
        "Test if the given point is internal to the polygon."
        return "bool"

    @PYB11const
    def convexContains(self,
                       point = "const Vector&",
                       countBoundary = ("const bool", "true"),
                       tol = ("const double", "1.0e-8")):
        "Test if the given point is internal to the polygon (assumes convexity)."
        return "bool"

    @PYB11const
    def intersect(self,
                  rhs = "const Polygon&"):
        "Test if we intersect another polygon."
        return "bool"

    @PYB11const
    def convexIntersect(self,
                        rhs = "const Polygon&"):
        "Test if we intersect another polygon (assumes convexity)"
        return "bool"

    @PYB11const
    def intersect(self,
                  rhs = "const std::pair<Vector, Vector>&"):
        "Test if we intersect a box represented by a min/max pair of coordinates."
        return "bool"

    @PYB11const
    def intersect(self,
                  x0 = "const Vector&",
                  x1 = "const Vector&"):
        "Return the intersections of this polygon with a line segment denoted by it's end points."
        return "std::vector<Vector>"

    @PYB11const
    def edges(self):
        "Get the edges as integer (node) pairs."
        return "std::vector<std::pair<unsigned, unsigned> >"

    @PYB11const
    def facetVertices(self):
        "Spit out a vector<vector<unsigned> > that encodes the facets."
        return "std::vector<std::vector<unsigned> >"

    def reconstruct(self,
                    vertices = "const std::vector<Vector>&",
                    facetVertices = "const std::vector<std::vector<unsigned> >&"):
        """Reconstruct the internal data given a set of verticies and the vertex
indices that define the facets."""
        return "void"

    @PYB11const
    def closestFacet(self, p = "const Vector&"):
        "Find the facet closest to the given point."
        return "unsigned"

    @PYB11const
    def distance(self, p="const Vector&"):
        "Compute the minimum distance to a point."
        return "double"

    @PYB11const
    def closestPoint(self, p="const Vector&"):
        "Find the point in the polygon closest to the given point."
        return "Vector"

    @PYB11const
    def convex(self):
        "Test if the polygon is convex"
        return "bool"

    def setBoundingBox(self):
        "Set the internal bounding box"
        return "void"

    #...........................................................................
    # Operators
    def __iadd__(self, rhs="Vector()"):
        return

    def __isub__(self, rhs="Vector()"):
        return

    def __add__(self, rhs="Vector()"):
        return

    def __sub__(self, rhs="Vector()"):
        return

    def __imul__(self, rhs="double()"):
        return
    
    def __idiv__(self, rhs="double()"):
        return
    
    def __mul__(self, rhs="double()"):
        return
    
    def __div__(self, rhs="double()"):
        return
    
    def __eq__(self):
        return

    def __ne__(self):
        return

    #...........................................................................
    # Properties
    centroid = PYB11property("Vector")
    vertices = PYB11property("std::vector<Vector>&", returnpolicy="reference_internal")
    facets = PYB11property("std::vector<Facet>&", returnpolicy="reference_internal")
    vertexUnitNorms = PYB11property("std::vector<Vector>&", returnpolicy="reference_internal")
    vertexFacetConnectivity = PYB11property("std::vector<std::vector<unsigned> >&", returnpolicy="reference_internal")
    facetFacetConnectivity = PYB11property("std::vector<std::vector<unsigned> >&", returnpolicy="reference_internal")
    xmin = PYB11property("const Vector&", returnpolicy="reference_internal")
    xmax = PYB11property("const Vector&", returnpolicy="reference_internal")
    volume = PYB11property("double")
