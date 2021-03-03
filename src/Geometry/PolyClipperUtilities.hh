//---------------------------------Spheral++----------------------------------//
// PolyClipperUtilities
//
// Methods for talking with PolyClipper:
//       https://github.com/LLNL/PolyClipper
//
// Created by JMO, Thu Feb  4 16:24:44 PST 2021
//----------------------------------------------------------------------------//
#ifndef __PolyClipperUtilities__
#define __PolyClipperUtilities__

#include "Geometry/Dimension.hh"

#include "polyclipper2d.hh"
#include "polyclipper3d.hh"

#include <vector>

namespace Spheral {

//------------------------------------------------------------------------------
// Adapter to use Spheral::GeomVector natively in PolyClipper
//------------------------------------------------------------------------------
template<int nDim>
struct GeomVectorAdapter {
  using VECTOR = typename Dim<nDim>::Vector;
  static VECTOR Vector(double a, double b)                   { return VECTOR(a, b); }    // only 2D
  static VECTOR Vector(double a, double b, double c)         { return VECTOR(a, b, c); } // only 3D
  static bool equal(const VECTOR& a, const VECTOR& b)        { return a == b; }
  static double& x(VECTOR& a)                                { return a[0]; }
  static double& y(VECTOR& a)                                { return a[1]; }
  static double& z(VECTOR& a)                                { return a[2]; }
  static double  x(const VECTOR& a)                          { return a[0]; }
  static double  y(const VECTOR& a)                          { return a[1]; }
  static double  z(const VECTOR& a)                          { return a[2]; }
  static double  dot(const VECTOR& a, const VECTOR& b)       { return a.dot(b); }
  static double  crossmag(const VECTOR& a, const VECTOR& b)  { return a.cross(b).z(); }  // only 2D
  static VECTOR  cross(const VECTOR& a, const VECTOR& b)     { return a.cross(b); }      // only 3D
  static double  magnitude2(const VECTOR& a)                 { return a.magnitude2(); }
  static double  magnitude(const VECTOR& a)                  { return a.magnitude(); }
  static VECTOR& imul(VECTOR& a, const double b)             { a *= b; return a; }
  static VECTOR& idiv(VECTOR& a, const double b)             { a /= b; return a; }
  static VECTOR& iadd(VECTOR& a, const VECTOR& b)            { a += b; return a; }
  static VECTOR& isub(VECTOR& a, const VECTOR& b)            { a -= b; return a; }
  static VECTOR  mul(const VECTOR& a, const double b)        { return a * b; }
  static VECTOR  div(const VECTOR& a, const double b)        { return a / b; }
  static VECTOR  add(const VECTOR& a, const VECTOR& b)       { return a + b; }
  static VECTOR  sub(const VECTOR& a, const VECTOR& b)       { return a - b; }
  static VECTOR  neg(const VECTOR& a)                        { return -a; }
  static VECTOR  unitVector(const VECTOR& a)                 { return a.unitVector(); }
  static std::string str(const VECTOR& a)                    { std::ostringstream os; os << a; return os.str(); }
};

//------------------------------------------------------------------------------
// Polygons
//------------------------------------------------------------------------------
using PolyClipperVertex2d = PolyClipper::Vertex2d<GeomVectorAdapter<2>>;
using PolyClipperPlane2d = PolyClipper::Plane<GeomVectorAdapter<2>>;
using PolyClipperPolygon = std::vector<PolyClipperVertex2d>;

void convertToPolyClipper(PolyClipperPolygon& polygon,
                          const Dim<2>::FacetedVolume& Spheral_polygon);

std::vector<std::set<int>> convertFromPolyClipper(Dim<2>::FacetedVolume& Spheral_polygon,
                                                  const PolyClipperPolygon& polygon);

//------------------------------------------------------------------------------
// Polyhedra
//------------------------------------------------------------------------------
using PolyClipperVertex3d = PolyClipper::Vertex3d<GeomVectorAdapter<3>>;
using PolyClipperPlane3d = PolyClipper::Plane<GeomVectorAdapter<3>>;
using PolyClipperPolyhedron = std::vector<PolyClipperVertex3d>;

void convertToPolyClipper(PolyClipperPolyhedron& polyhedron,
                          const Dim<3>::FacetedVolume& Spheral_polyhedron);

std::vector<std::set<int>> convertFromPolyClipper(Dim<3>::FacetedVolume& Spheral_polyhedron,
                                                  const PolyClipperPolyhedron& polyhedron);

}

#endif
