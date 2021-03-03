from PYB11Generator import *

class PolyClipperVertex2d:
    '''The PolyClipper 2D (x,y) vertex type.

Vertex2d is used to encode Polygons, so that PolyClipper Polygons are simply 
collections of vertices.  A Vertex2d consists of a position and a pair of 
neighbor Vertex2d indices (a,b), which represent the (previous, next) vertices
going around a polygon in counterclockwise order.

New vertices are created by clipping operations, and Vertex maintains a set of 
the plane IDs that have touched/clipped this vertex.

Vertex2d also maintains a "comp" and "ID" attributes, which are primarily for
internal usage during PolyClipper clipping operations.
'''

    PYB11typedefs = """
  typedef Dim<2>::Vector Vector;
"""

    #---------------------------------------------------------------------------
    # Constructors
    #---------------------------------------------------------------------------
    def pyinit0(self):
        "Default constructor"

    def pyinit1(self,
                pos = "const Vector&"):
        "Construct with a position"

    def pyinit2(self,
                pos = "const Vector&",
                c = "const int"):
        "Construct with a position and initial 'comp' value"

    def pyinit3(self,
                rhs = "const PolyClipperVertex2d&"):
        "Copy constructor"

    #---------------------------------------------------------------------------
    # Operators
    #---------------------------------------------------------------------------
    def __eq__(self):
        return

    #---------------------------------------------------------------------------
    # Methods
    #---------------------------------------------------------------------------
    @PYB11implementation('''
[](const PolyClipperVertex2d& self) { 
  auto result = "{pos=(" + std::to_string(self.position.x()) + " " + std::to_string(self.position.y()) + 
                "), neighbors=(" + std::to_string(self.neighbors.first) + " " + std::to_string(self.neighbors.second) +
                "), ID=" + std::to_string(self.ID) +
                ", clips=( ";
  for (const auto x: self.clips) result += std::to_string(x) + " ";
  result += ")}";
  return result;
}''')
    def __repr__(self):
        return

    #---------------------------------------------------------------------------
    # Attributes
    #---------------------------------------------------------------------------
    position = PYB11readwrite()
    neighbors = PYB11readwrite()
    comp = PYB11readwrite()
    ID = PYB11readwrite()
    clips = PYB11readwrite()
