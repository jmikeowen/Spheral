from PYB11Generator import *

@PYB11namespace("PolyClipper")
@PYB11template("VA")
class Plane:
    """The PolyClipper plane.

A plane is defined by a signed scalar distance (dist) and unit normal (normal).
The distance d represents the signed shortest distance from the plane to the 
origin:

     dist = -P0.dot(normal),

where "P0" is a point in the plane.  Note the normal defines the "above" and 
"below" conventions for a plane, so for any point "p" with distance "s" from the 
plane:

     s = (p - P0).dot(normal)

s < 0 implies p is below the plane, s > 0 above, and s == 0 means p is in the 
plane.

Planes also optionally keep an integer ID, which is used during clipping 
operations to track which plane(s) are responsible for each vertex."""

    PYB11typedefs = """
    using Vector = typename %(VA)s::VECTOR;
"""

    #---------------------------------------------------------------------------
    # Constructors
    #---------------------------------------------------------------------------
    def pyinit0(self):
        "Default constructor"

    def pyinit1(self,
                d = "const double",
                nhat = "const Vector"):
        "Construct with (signed distance, unit normal) = (d, nhat)"

    def pyinit2(self,
                p = "const Vector&",
                nhat = "const Vector&"):
        "Construct using a point in the plane (p) and unit normal (nhat)"

    def pyinit3(self,
                p = "const Vector&",
                nhat = "const Vector&",
                id = "const int"):
        "Construct using a point in the plane (p), unit normal (nhat), plane ID (id)"

    def pyinit4(self,
                rhs = "const Plane<%(VA)s>&"):
        "Copy constructor"
        
    #---------------------------------------------------------------------------
    # Operators
    #---------------------------------------------------------------------------
    def __eq__(self):
        return

    def __ne__(self):
        return

    def __lt__(self):
        return

    def __gt__(self):
        return

    #---------------------------------------------------------------------------
    # Methods
    #---------------------------------------------------------------------------
    @PYB11implementation('''[](const Plane<%(VA)s>& self) { return "{" + std::to_string(self.dist) + ", (" + std::to_string(self.normal.x()) + ", " + std::to_string(self.normal.y()) + ", " + std::to_string(self.normal.z()) + ")}"; }''')
    def __repr__(self):
        return

    #---------------------------------------------------------------------------
    # Attributes
    #---------------------------------------------------------------------------
    dist = PYB11readwrite()
    normal = PYB11readwrite()
