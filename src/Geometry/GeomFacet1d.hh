//---------------------------------Spheral++----------------------------------//
// GeomFacet1d -- A facet of a Box1d (a point and a normal)
//----------------------------------------------------------------------------//
#ifndef __Spheral_GeomFacet1d__
#define __Spheral_GeomFacet1d__

#include <vector>
#include <iostream>

#include "Geometry/GeomVector_fwd.hh"

namespace Spheral {

// Forward declare the polygon.
class GeomPolygon;

class GeomFacet1d {
public:
  //--------------------------- Public Interface ---------------------------//
  typedef GeomVector<1> Vector;

  //----------------------------------------------------------------------------
  // Constructors, assignment, destructor.
  //----------------------------------------------------------------------------
  GeomFacet1d();
  GeomFacet1d(const Vector& position,
              const Vector& normal);
  GeomFacet1d(const GeomFacet1d& rhs);
  GeomFacet1d& operator=(const GeomFacet1d& rhs);
  ~GeomFacet1d();
  
  // Access the point.
  const Vector& point() const;

  // Access the normal.
  const Vector& normal() const;

  // For consistency with GeomFacet3d.
  void decompose(std::vector<std::array<Vector, 1>>& subfacets) const;
  
private:
  //--------------------------- Private Interface ---------------------------//
  Vector mPoint;
  Vector mNormal;
};

// Provide an ostream operator for GeomFacet1d.
std::ostream& operator <<(std::ostream& os, const GeomFacet1d& facet);

}

#include "GeomFacet1dInline.hh"

#else 

namespace Spheral {
  class GeomFacet1d;
}

#endif

