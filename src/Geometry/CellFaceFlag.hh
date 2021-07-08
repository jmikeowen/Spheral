//------------------------------------------------------------------------------
// A struct for encoding cell face information returned by computeVoronoiVolume
//------------------------------------------------------------------------------
#ifndef __Spheral_CellFaceFlag__
#define __Spheral_CellFaceFlag__

#include "Utilities/DataTypeTraits.hh"
#include "Utilities/packElement.hh"
#include "Utilities/DBC.hh"

#include <iostream>

namespace Spheral {

struct CellFaceFlag {
  int cellFace;  // The index of the face in the cell.facets array
  int nodeListj; // The NodeList of the opposite node across the face
  int j;         // The index of the opposite node across the face
  CellFaceFlag():
    cellFace(-1),
    nodeListj(-1),
    j(-1) {}
  CellFaceFlag(int x, int y, int z):
    cellFace(x),
    nodeListj(y),
    j(z) {}
  bool operator==(const CellFaceFlag& rhs) const {
    return (cellFace == rhs.cellFace and
            nodeListj == rhs.nodeListj and
            j == rhs.j);
  }
};

//------------------------------------------------------------------------------
// Communication/serialization traits
//------------------------------------------------------------------------------
template<>
struct DataTypeTraits<CellFaceFlag> {
  typedef int ElementType;
  static bool fixedSize() { return true; }
  static int numElements(const CellFaceFlag&) { return 3; }
  static CellFaceFlag zero() { return CellFaceFlag({-1, -1, -1}); }
  using AxomType = int;
};

template<>
inline
void packElement<CellFaceFlag>(const CellFaceFlag& value,
                                std::vector<char>& buffer) {
  packElement(value.cellFace, buffer);
  packElement(value.nodeListj, buffer);
  packElement(value.j, buffer);
}

template<>
inline
void unpackElement<CellFaceFlag>(CellFaceFlag& value,
                                 std::vector<char>::const_iterator& itr,
                                 const std::vector<char>::const_iterator& endPackedVector) {
  unpackElement(value.cellFace, itr, endPackedVector);
  unpackElement(value.nodeListj, itr, endPackedVector);
  unpackElement(value.j, itr, endPackedVector);
  ENSURE(itr <= endPackedVector);
}

//------------------------------------------------------------------------------
// Output (ostream) operator.
//------------------------------------------------------------------------------
inline
std::ostream&
operator<<(std::ostream& os, const CellFaceFlag& x) {
  os << "(" << x.cellFace << " " << x.nodeListj << " " << x.j << ")";
  return os;
}

}

#endif
