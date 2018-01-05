//---------------------------------PolyClipper--------------------------------//
// Clip a faceted volume (polygon or polyhedron) by a set of planes in place.
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

#include "polyclipper.hh"
#include "polyclipper_utilities.hh"
#include "Utilities/DBC.hh"
#include "Utilities/Timer.hh"

#include <list>
#include <map>
#include <iostream>
#include <iterator>
#include <algorithm>

// Declare the timers.
extern Timer TIME_PC3d_convertto;
extern Timer TIME_PC3d_convertfrom;
extern Timer TIME_PC3d_copy;
extern Timer TIME_PC3d_moments;
extern Timer TIME_PC3d_clip;
extern Timer TIME_PC3d_planes;
extern Timer TIME_PC3d_checkverts;
extern Timer TIME_PC3d_insertverts;
extern Timer TIME_PC3d_planeverts;
extern Timer TIME_PC3d_linknew;
extern Timer TIME_PC3d_compress;

namespace PolyClipper {

using namespace std;

namespace {    // anonymous methods

//------------------------------------------------------------------------------
// Compare a plane and point (our built-in plane one has some issues).
//------------------------------------------------------------------------------
inline
int compare(const Spheral::Dim<3>::Vector& planePoint,
            const Spheral::Dim<3>::Vector& planeNormal,
            const Spheral::Dim<3>::Vector& point) {
  const auto sgndist = planeNormal.dot(point - planePoint);
  if (std::abs(sgndist) < 1.0e-10) return 0;
  return sgn0(sgndist);
}

//------------------------------------------------------------------------------
// Intersect a line-segment with a plane.
//------------------------------------------------------------------------------
inline
Spheral::Dim<3>::Vector
segmentPlaneIntersection(const Spheral::Dim<3>::Vector& a,       // line-segment begin
                         const Spheral::Dim<3>::Vector& b,       // line-segment end
                         const Spheral::Dim<3>::Vector& p,       // point in plane
                         const Spheral::Dim<3>::Vector& phat) {  // plane unit normal

  const auto ab = b - a;
  if (fuzzyEqual(ab.magnitude(), 0.0, 1.0e-10)) return a;
  const auto abhat = ab.unitVector();
  CHECK2(std::abs(abhat.dot(phat)) > 0.0, (abhat.dot(phat)) << " " << a << " " << b << " " << abhat << " " << phat);
  const auto s = std::max(0.0, std::min(ab.magnitude(), (p - a).dot(phat)/(abhat.dot(phat))));
  CHECK2(s >= 0.0 and s <= ab.magnitude(), s << " " << ab.magnitude());
  const auto result = a + s*abhat;
  // CHECK2(fuzzyEqual((result - p).dot(phat), 0.0, 1.0e-10),
  //        a << " " << b << " " << s << " " << result << " " << (result - p).dot(phat));
  return result;
}

//------------------------------------------------------------------------------
// Find the next neighbor in CCW order in the neighbor set of a vertex.
// This should be the previous vertex from our entry value.
//------------------------------------------------------------------------------
inline
Vertex3d*
nextInFaceLoop(Vertex3d* vptr, Vertex3d* vprev) {
  const auto itr = find(vptr->neighbors.begin(), vptr->neighbors.end(), vprev);
  CHECK(itr != vptr->neighbors.end());
  if (itr == vptr->neighbors.begin()) {
    return vptr->neighbors.back();
  } else {
    return *(itr - 1);
  }
}

}              // anonymous methods

//------------------------------------------------------------------------------
// Return the vertices ordered in faces.
// Implicitly uses the convention that neighbors for each vertex are arranged
// counter-clockwise viewed from the exterior.
//------------------------------------------------------------------------------
vector<vector<const Vertex3d*>>
extractFaces(const Polyhedron& poly) {

  typedef pair<const Vertex3d*, const Vertex3d*> Edge;
  typedef vector<const Vertex3d*> Face;

  // Prepare the result.
  vector<vector<const Vertex3d*>> result;

  // {
  //   int i = 0;
  //   for (const auto& v: poly) v.ID = i++;
  //   for (const auto& v: poly) {
  //     cerr << " **> " << v.ID << " " << v.position << " :";
  //     for (const auto nptr: v.neighbors) cerr << " " << nptr->ID;
  //     cerr << endl;
  //   }
  // }

  // Walk each vertex in the polyhedron.
  set<Edge> edgesWalked;
  for (const auto& v: poly) {
    if (v.comp >= 0) {

      // {
      //   cerr << " --> " << v.ID << " " << v.position << " :";
      //   for (const auto nptr: v.neighbors) cerr << " " << nptr->ID;
      //   cerr << endl;
      // }

      // Check every (outgoing) edge attached to this vertex.
      for (const auto nptr: v.neighbors) {
        CHECK(nptr->comp >= 0);

        // Has this edge been walked yet?
        if (edgesWalked.find(make_pair(nptr, &v)) == edgesWalked.end()) {
          Face face(1, nptr);
          const Vertex3d* vstart = nptr;
          const Vertex3d* vnext = &v;
          const Vertex3d* vprev = nptr;
          // cerr << "Face: " << nptr->ID;

          // Follow around the face represented by this edge until we get back
          // to our starting vertex.
          while (vnext != vstart) {
            // cerr << " " << vnext->ID;
            face.push_back(vnext);
            CHECK(edgesWalked.find(make_pair(vprev, vnext)) == edgesWalked.end());
            edgesWalked.insert(make_pair(vprev, vnext));
            auto itr = find(vnext->neighbors.begin(), vnext->neighbors.end(), vprev);
            CHECK(itr != vnext->neighbors.end());
            vprev = vnext;
            if (itr == vnext->neighbors.begin()) {
              vnext = vnext->neighbors.back();
            } else {
              vnext = *(itr - 1);
            }
          }
          // cerr << endl;
          edgesWalked.insert(make_pair(vprev, vnext));   // Final edge connecting last->first vertex
          CHECK(face.size() >= 3);
          result.push_back(face);

        }
      }
    }
  }

  // Post-conditions.
  BEGIN_CONTRACT_SCOPE
  {
    // Every pair should have been walked twice, once in each direction.
    for (const auto& v: poly) {
      for (const auto nptr: v.neighbors) {
        CHECK(edgesWalked.find(make_pair(&v, nptr)) != edgesWalked.end());
        CHECK(edgesWalked.find(make_pair(nptr, &v)) != edgesWalked.end());
      }
    }
  }
  END_CONTRACT_SCOPE

  // That's it.
  return result;
}

//------------------------------------------------------------------------------
// Return a nicely formatted string representing the polyhedron.
//------------------------------------------------------------------------------
std::string
polyhedron2string(const Polyhedron& poly) {

  // First assign ID's to the vertices.
  // This is why ID is mutable.
  {
    auto i = 0;
    for (const auto& v: poly) v.ID = i++;
  }

  ostringstream s;
  for (const auto& v: poly) {
    s << "ID=" << v.ID << " comp=" << v.comp << " @ " << v.position
      << " neighbors=[";
    for (const auto nptr: v.neighbors) s << " " << nptr->ID;
    s << "]\n";
  }

  return s.str();
}

// std::string
// polyhedron2string(const Polyhedron& poly) {
//   ostringstream s;
//   s << "[";

//   // Get the vertices in face ordering.
//   const vector<vector<const Vertex3d*>> faces;

//   // Now output the face vertex coordinates.
//   for (const auto& face: faces) {
//     s << "[";
//     for (const auto vptr: face) {
//       s << " " << vptr->position;
//     }
//     s << "]\n ";
//   }
//   s << "]";

//   return s.str();
// }

//------------------------------------------------------------------------------
// Convert Spheral::GeomPolyhedron -> PolyClipper::Polyhedron.
//------------------------------------------------------------------------------
void convertToPolyhedron(Polyhedron& polyhedron,
                      const Spheral::Dim<3>::FacetedVolume& Spheral_polyhedron) {
  TIME_PC3d_convertto.start();

  const auto& vertPositions = Spheral_polyhedron.vertices();
  const auto& facets = Spheral_polyhedron.facets();
  const auto  nverts = vertPositions.size();
  const auto  nfacets = facets.size();

  // Build the PolyClipper Vertex3d's, but without connectivity yet.
  polyhedron.clear();
  vector<Vertex3d*> id2vert;
  for (auto k = 0; k < nverts; ++k) {
    polyhedron.push_back(Vertex3d(vertPositions[k], 1));
    id2vert.push_back(&polyhedron.back());
  }
  CHECK(id2vert.size() == nverts);

  // Note all the edges associated with each vertex.
  vector<vector<pair<int, int>>> vertexPairs(nverts);
  int v0, vprev, vnext;
  for (const auto& facet: facets) {
    const auto& ipoints = facet.ipoints();
    const int   n = ipoints.size();
    for (int k = 0; k < n; ++k) {
      v0 = ipoints[k];
      vprev = ipoints[(k - 1 + n) % n];
      vnext = ipoints[(k + 1 + n) % n];
      vertexPairs[v0].push_back(make_pair(vnext, vprev));   // Gotta reverse!
    }
  }

  // Now we can build vertex->vertex connectivity.
  for (auto k = 0; k < nverts; ++k) {

    // Sort the edges associated with this vertex.
    const auto n = vertexPairs[k].size();
    CHECK(n >= 3);
    for (auto i = 0; i < n - 1; ++i) {
      auto j = i + 1;
      while (j < n and vertexPairs[k][i].second != vertexPairs[k][j].first) ++j;
      CHECK(j < n);
      swap(vertexPairs[k][i + 1], vertexPairs[k][j]);
    }

    // Now that they're in order, create the neighbors.
    for (auto i = 0; i < n; ++i) {
      id2vert[k]->neighbors.push_back(id2vert[vertexPairs[k][i].first]);
    }
    CHECK(id2vert[k]->neighbors.size() == vertexPairs[k].size());
  }

  CHECK(polyhedron.size() == nverts);
  TIME_PC3d_convertto.stop();
}

//------------------------------------------------------------------------------
// Convert PolyClipper::Polyhedron -> Spheral::GeomPolyhedron.
//------------------------------------------------------------------------------
void convertFromPolyhedron(Spheral::Dim<3>::FacetedVolume& Spheral_polyhedron,
                           const Polyhedron& polyhedron) {
  TIME_PC3d_convertfrom.start();

  // Useful types.
  typedef Spheral::Dim<3>::FacetedVolume FacetedVolume;
  typedef Spheral::Dim<3>::Vector Vector;

  if (polyhedron.empty()) {

    Spheral_polyhedron = FacetedVolume();

  } else {

    // extractFaces actually does most of the work.
    const auto faces = extractFaces(polyhedron);

    // Number and extract the active vertices.
    vector<Vector> coords;
    {
      int i = 0;
      for (auto itr = polyhedron.begin(); itr != polyhedron.end(); ++itr) {
        if (itr->comp >= 0) {
          coords.push_back(itr->position);
          itr->ID = i++;
        }
      }
    }
    CHECK(coords.size() == count_if(polyhedron.begin(), polyhedron.end(),
                                    [](const Vertex3d& x) { return x.comp >= 0; }));

    // Extract the faces as integer vertex index loops.
    vector<vector<unsigned>> facets(faces.size());
    for (auto k = 0; k < faces.size(); ++k) {
      facets[k].resize(faces[k].size());
      transform(faces[k].begin(), faces[k].end(), facets[k].begin(),
                [](const Vertex3d* vptr) { return vptr->ID; });
    }

    // Now we can build the Spheral::Polyhedron.
    Spheral_polyhedron = FacetedVolume(coords, facets);

  }
  TIME_PC3d_convertfrom.stop();
}

//------------------------------------------------------------------------------
// Copy a PolyClipper::Polyhedron.
//------------------------------------------------------------------------------
void copyPolyhedron(Polyhedron& polyhedron,
                 const Polyhedron& polyhedron0) {
  TIME_PC3d_copy.start();
  polyhedron.clear();
  if (not polyhedron0.empty()) {
    std::map<const Vertex3d*, Vertex3d*> ptrMap;
    for (auto& v: polyhedron0) {
      polyhedron.push_back(v);
      ptrMap[&v] = &polyhedron.back();
    }
    for (auto& v: polyhedron) {
      transform(v.neighbors.begin(), v.neighbors.end(), v.neighbors.begin(),
                [&](const Vertex3d* vptr) { return ptrMap[vptr]; });
    }
  }
  TIME_PC3d_copy.stop();
}

//------------------------------------------------------------------------------
// Compute the zeroth and first moment of a Polyhedron.
//------------------------------------------------------------------------------
void moments(double& zerothMoment, Spheral::Dim<3>::Vector& firstMoment,
             const Polyhedron& polyhedron) {
  TIME_PC3d_moments.start();

  // Useful types.
  typedef Spheral::Dim<3>::Vector Vector;

  // Clear the result for accumulation.
  zerothMoment = 0.0;
  firstMoment = Vector::zero;

  if (not polyhedron.empty()) {

    // Walk the polyhedron, and add up our results face by face.
    const auto faces = extractFaces(polyhedron);
    double dV;
    for (const auto& face: faces) {
      const auto nverts = face.size();
      CHECK(nverts >= 3);
      const auto& v0 = face[0]->position;
      for (auto i = 1; i < nverts - 1; ++i) {
        const auto& v1 = face[i]->position;
        const auto& v2 = face[i+1]->position;
        dV = v0.dot(v1.cross(v2));
        zerothMoment += dV;
        firstMoment += dV*(v0 + v1 + v2);
      }
    }
    zerothMoment /= 6.0;
    firstMoment *= Spheral::safeInv(24.0*zerothMoment);
  }
  TIME_PC3d_moments.stop();
}

//------------------------------------------------------------------------------
// Clip a polyhedron by planes.
//------------------------------------------------------------------------------
void clipPolyhedron(Polyhedron& polyhedron,
                 const std::vector<Spheral::GeomPlane<Spheral::Dim<3>>>& planes) {
  TIME_PC3d_clip.start();

  // Useful types.
  typedef Spheral::Dim<3>::Vector Vector;

  // cerr << "Initial:\n" << polyhedron2string(polyhedron) << endl;

  // Loop over the planes.
  TIME_PC3d_planes.start();
  auto kplane = 0;
  const auto nplanes = planes.size();
  while (kplane < nplanes and not polyhedron.empty()) {
    const auto& plane = planes[kplane++];
    const auto& p0 = plane.point();
    const auto& phat = plane.normal();
    // cerr << "Clip plane: " << p0 << " " << phat << endl;

    // Check the current set of vertices against this plane.
    // Also keep track of any vertices that landed exactly in-plane.
    TIME_PC3d_checkverts.start();
    auto above = true;
    auto below = true;
    vector<Vertex3d*> planeVertices;
    for (auto& v: polyhedron) {
      const auto barf = ((v.position - Vector( 3.46945e-18, 0.0657523, -0.0728318 )).magnitude() < 1.0e-3);
      if (barf) cerr << v.position << " " << v.comp << " " << v.neighbors.size() << " : " << kplane << endl;
      v.comp = compare(p0, phat, v.position);
      if (v.comp == 1) {
        below = false;
      } else if (v.comp == -1) {
        above = false;
      } else {
        CHECK(v.comp == 0);
        planeVertices.push_back(&v);
      }
    }
    CHECK(not (above and below));
    TIME_PC3d_checkverts.stop();

    // Did we get a simple case?
    if (below) {
      // The polyhedron is entirely below the clip plane, and is therefore entirely removed.
      // No need to check any more clipping planes -- we're done.
      polyhedron.clear();

    } else if (not above) {

      // This plane passes through the polyhedron.
      // Insert any new vertices.
      TIME_PC3d_insertverts.start();
      vector<Vertex3d*> newVertices;
      const auto nverts0 = polyhedron.size();
      {

        // Look for any new vertices we need to insert.
        auto i = 0;
        for (auto vitr = polyhedron.begin(); i < nverts0; ++vitr, ++i) {   // Only check vertices before we start adding new ones.
          auto& v = *vitr;
          const auto barf = ((v.position - Vector( 3.46945e-18, 0.0657523, -0.0728318 )).magnitude() < 1.0e-3);
          if (v.comp == 1) {

            // This vertex survives clipping -- check the neighbors.
            const auto nneigh = v.neighbors.size();
            CHECK(nneigh >= 3);
            for (auto j = 0; j < nneigh; ++j) {
              auto nptr = v.neighbors[j];
              if (nptr->comp == -1) {

                // This edge straddles the clip plane, so insert a new vertex.
                polyhedron.push_back(Vertex3d(segmentPlaneIntersection(v.position,
                                                                       nptr->position,
                                                                       p0,
                                                                       phat),
                                              2));         // 2 indicates new vertex
                polyhedron.back().neighbors = {nptr, &v};
                auto itr = find(nptr->neighbors.begin(), nptr->neighbors.end(), &v);
                CHECK(itr != nptr->neighbors.end());
                *itr = &polyhedron.back();
                v.neighbors[j] = &polyhedron.back();
                newVertices.push_back(&polyhedron.back());
                // cerr << " --> Inserting new vertex @ " << polyhedron.back().position << endl;

              }
            }
          }
        }
      }
      const auto nverts = polyhedron.size();
      // cerr << "After insertion:\n" << polyhedron2string(polyhedron) << endl;
      TIME_PC3d_insertverts.stop();

      // Next handle reconnecting any vertices that were exactly in-plane.
      TIME_PC3d_planeverts.start();
      for (auto vptr: planeVertices) {
        CHECK(vptr->comp == 0);
        const auto nneigh = vptr->neighbors.size();
        CHECK(nneigh >= 3);
        for (auto j = 0; j < nneigh; ++j) {
          auto nptr = vptr->neighbors[j];
          if (nptr->comp == -1) {

            // Yep, this edge is clipped, so look for where the in-plane vertex should hook up.
            auto vprev = vptr;
            auto vnext = nptr;
            auto tmp = vnext;
            auto k = 0;
            while (vnext->comp == -1 and k++ < nverts) {
              tmp = vnext;
              vnext = nextInFaceLoop(vnext, vprev);
              vprev = tmp;
            }
            CHECK(vprev->comp == -1);
            CHECK(vnext->comp != -1);
            CHECK(vnext != vptr);
            vptr->neighbors[j] = vnext;

            const auto barf = ((vnext->position - Vector( 3.46945e-18, 0.0657523, -0.0728318 )).magnitude() < 1.0e-3);
            if (barf) cerr << "Deg: " << vptr->position << " " << nneigh << endl;

            // Figure out which pointer on the new neighbor should point back at vptr.
            auto itr = find(vnext->neighbors.begin(), vnext->neighbors.end(), vprev);
            CHECK(itr != vnext->neighbors.end());
            *itr = vptr;
          }
        }
      }
      TIME_PC3d_planeverts.stop();

      // For each new vertex, link to the neighbors that survive the clipping.
      TIME_PC3d_linknew.start();
      // vector<pair<Vertex3d*, Vertex3d*>> newEdges;
      for (auto vptr: newVertices) {
        CHECK(vptr->comp == 2);

        // Look for any neighbors of the vertex that are clipped.
        const auto nneigh = vptr->neighbors.size();

        const auto barf = ((vptr->position - Vector( 3.46945e-18, 0.0657523, -0.0728318 )).magnitude() < 1.0e-3);
        if (barf) cerr << "New: " << vptr->position << " " << nneigh << endl;

        for (auto j = 0; j < nneigh; ++j) {
          auto nptr = vptr->neighbors[j];
          if (barf) cerr << " **> " << nptr->position << " " << nptr->comp << " -> ";
          if (nptr->comp == -1) {

            // This neighbor is clipped, so look for the first unclipped vertex along this face loop.
            auto vprev = vptr;
            auto vnext = nptr;
            auto tmp = vnext;
            // cerr << vptr->ID << ": ( " << vprev->ID << " " << vnext->ID << ")";
            auto k = 0;
            while (vnext->comp == -1 and k++ < nverts) {
              tmp = vnext;
              vnext = nextInFaceLoop(vnext, vprev);
              vprev = tmp;
              // cerr << " (" << vprev->ID << " " << vnext->ID << ")";
            }
            // cerr << endl;
            CHECK(vnext->comp != -1);
            vptr->neighbors[j] = vnext;
            vnext->neighbors.insert(vnext->neighbors.begin(), vptr);
            // newEdges.push_back(make_pair(vnext, vptr));
            if (barf) cerr << vnext->position << " " << vnext->comp << " " << vnext->neighbors.size();
          }
          if (barf) cerr << endl;
        }
      }
      // const auto nNewEdges = newEdges.size();
      // CHECK(nNewEdges >= 3);
      TIME_PC3d_linknew.stop();

      // // Collapse any degenerate vertices back onto the originals.
      // TIME_PC3d_degenerate.start();
      // {
      //   Vertex3d *vdeg, *vptr;
      //   for (auto& vpair: degenerateVertices) {
      //     tie(vdeg, vptr) = vpair;
      //     vptr->neighbors.reserve(vptr->neighbors.size() + vdeg->neighbors.size());
      //     vptr->neighbors.insert(vptr->neighbors.end(), vdeg->neighbors.begin(), vdeg->neighbors.end());
      //     vdeg->comp = -1;
      //     for (auto& v: polyhedron) { // Get rid of this!
      //       replace(v.neighbors.begin(), v.neighbors.end(), vdeg, vptr);
      //     }
      //   }
      // }
      // TIME_PC3d_degenerate.stop();

      // Remove the clipped vertices, compressing the polyhedron.
      TIME_PC3d_compress.start();
      for (auto vitr = polyhedron.begin(); vitr != polyhedron.end();) {
        if (vitr->comp < 0) {
          vitr = polyhedron.erase(vitr);
        } else {
          if (!(vitr->neighbors.size() >= 3)) {
            auto v = *vitr;
            cerr << " !! " << v.position << " " << v.comp << " " << v.neighbors.size() << " : " << endl;
            for (auto nptr: v.neighbors) {
              cerr << "    " << nptr->position << nptr->comp << " " << nptr->neighbors.size() << " " << (nptr->position - v.position).unitVector() << endl;
            }
          }
          CHECK(vitr->neighbors.size() >= 3);
          ++vitr;
        }
      }
      // cerr << "After compression:\n" << polyhedron2string(polyhedron) << endl;

      // Is the polyhedron gone?
      if (polyhedron.size() < 4) polyhedron.clear();
      TIME_PC3d_compress.stop();
    }
  }
  TIME_PC3d_planes.stop();
}

}