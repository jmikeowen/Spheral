from PYB11Generator import *

class PolyClipperVertex3d:
    '''The PolyClipper 3D (x,y,z) vertex type.

Vertex3d is used to encode Polyhedra, so that PolyClipper Polyhedra are simply 
collections of vertices.  A Vertex3d consists of a position and a vector of 
neighbor Vertex3d indices (a,b,c,...), which represent the connected vertices
to this one going counterclockwise around this vertex viewed from outside the
polyhedron.

New vertices are created by clipping operations, and Vertex maintains a set of 
the plane IDs that have touched/clipped this vertex.

Vertex3d also maintains a "comp" and "ID" attributes, which are primarily for
internal usage during PolyClipper clipping operations.
'''

    PYB11typedefs = """
  typedef Dim<3>::Vector Vector;
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
                rhs = "const PolyClipperVertex3d&"):
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
[](const PolyClipperVertex3d& self) { 
  auto result = "{pos=(" + std::to_string(self.position.x()) + " " + std::to_string(self.position.y()) + + " " + std::to_string(self.position.z()) + 
                "), neighbors=( ";
  for (const auto x: self.neighbors) result += std::to_string(x) + " "; 
  result += "), ID=" + std::to_string(self.ID) +
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
