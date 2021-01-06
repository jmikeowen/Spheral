#include "GeomVector.hh"
#include "Utilities/SpheralFunctions.hh"
#include <array>

namespace Spheral {

//------------------------------------------------------------------------------
// Constructors, destructor.
//------------------------------------------------------------------------------
// We really don't want the default constructor, but it's required to have 
// std::vectors of these.
inline
GeomFacet1d::
GeomFacet1d():
  mPoint(0.0),
  mNormal(1.0) {
  VERIFY(false);
}

inline
GeomFacet1d::
GeomFacet1d(const Vector& point,
            const Vector& normal):
  mPoint(point),
  mNormal(normal) {
}

inline
GeomFacet1d::
GeomFacet1d(const GeomFacet1d& rhs):
  mPoint(rhs.mPoint),
  mNormal(rhs.mNormal) {
}

inline
GeomFacet1d&
GeomFacet1d::
operator=(const GeomFacet1d& rhs) {
  if (this != &rhs) {
    mPoint = rhs.mPoint;
    mNormal = rhs.mNormal;
  }
  return *this;
}

inline
GeomFacet1d::
~GeomFacet1d() {
}

//------------------------------------------------------------------------------
// Access the points and normal.
//------------------------------------------------------------------------------
inline
const GeomFacet1d::Vector&
GeomFacet1d::
point() const {
  return mPoint;
}

inline
const GeomFacet1d::Vector&
GeomFacet1d::
normal() const {
  return mNormal;
}

//------------------------------------------------------------------------------
// This facet does not need to be decomposed.
//------------------------------------------------------------------------------
inline
void
GeomFacet1d::
decompose(std::vector<std::array<Vector, 1>>& subfacets) const {
  //subfacets[0] = std::array<Vector,1>{Vector(mPoint)};
  subfacets = {{mPoint}};
}

//------------------------------------------------------------------------------
// Output (ostream) operator.
//------------------------------------------------------------------------------
inline
std::ostream&
operator<<(std::ostream& os, const GeomFacet1d& facet) {
  os << "GeomFacet1d( point  : " << facet.point() << "\n"
     << "             normal : " << facet.normal() 
     << "\n)";
  return os;
}


}
