//---------------------------------PolyClipper--------------------------------//
// Clip a faceted volume (polygon or polyhedron) by a plane in place.
//
// We use the convention that any portion of the faceted volume "below" the 
// plane is clipped, i.e., only the portion of the faceted volume "above" the 
// plane
//    plane.compare(point) >= 0
// is retained.
//
// The algorithms herein are based on R3D as outlined in 
// Powell, D., & Abel, T. (2015). An exact general remeshing scheme applied to 
// physically conservative voxelization. Journal of Computational Physics, 297, 340–356.
//
// Created by J. Michael Owen, Tue Nov 28 10:00:51 PST 2017
//----------------------------------------------------------------------------//
#ifndef __PolyClipper_hh__
#define __PolyClipper_hh__

#include "Geometry/Dimension.hh"
#include "Geometry/GeomPlane.hh"

#include <list>

namespace PolyClipper {

//------------------------------------------------------------------------------
// The 2D vertex struct, which we use to encode polygons.
//------------------------------------------------------------------------------
struct Vertex2d {
  typedef Spheral::Dim<2>::Vector Vector;
  Vector position;
  std::pair<Vertex2d*, Vertex2d*> neighbors;
  int comp;
  Vertex2d():                               position(),    neighbors(), comp(1) {}
  Vertex2d(const Vector& pos):              position(pos), neighbors(), comp(1) {}
  Vertex2d(const Vector& pos, const int c): position(pos), neighbors(), comp(c) {}
};

//------------------------------------------------------------------------------
// 2D (polygon) methods.
//------------------------------------------------------------------------------
typedef std::list<Vertex2d> Polygon;

void convertToPolygon(Polygon& polygon,
                      const Spheral::Dim<2>::FacetedVolume& Spheral_polygon);

void convertFromPolygon(Spheral::Dim<2>::FacetedVolume& Spheral_polygon,
                        const Polygon& polygon);

void copyPolygon(Polygon& polygon,
                 const Polygon& polygon0);

void moments(double& zerothMoment, Spheral::Dim<2>::Vector& firstMoment,
             const Polygon& polygon);

void clipPolygon(Polygon& poly,
                 const std::vector<Spheral::GeomPlane<Spheral::Dim<2>>>& planes);

}

#endif
