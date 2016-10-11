//---------------------------------Spheral++------------------------------------
// Compute the volume per point based on the Voronoi tessellation.
//------------------------------------------------------------------------------
#include "computeVoronoiVolume.hh"
#include "Field/Field.hh"
#include "Field/FieldList.hh"
#include "NodeList/NodeList.hh"
#include "Utilities/PairComparisons.hh"

namespace Spheral {
namespace CRKSPHSpace {

using namespace std;

using FieldSpace::Field;
using FieldSpace::FieldList;
using NodeSpace::NodeList;
using NeighborSpace::ConnectivityMap;

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------
void
computeVoronoiVolume(const FieldList<Dim<1>, Dim<1>::Vector>& position,
                     const FieldList<Dim<1>, Dim<1>::SymTensor>& H,
                     const ConnectivityMap<Dim<1> >& connectivityMap,
                     const Dim<1>::Scalar kernelExtent,
                     FieldList<Dim<1>, int>& surfacePoint,
                     FieldList<Dim<1>, Dim<1>::Scalar>& vol) {

  const unsigned numGens = position.numNodes();
  const unsigned numNodeLists = position.size();

  typedef Dim<1>::Scalar Scalar;
  typedef Dim<1>::Vector Vector;
  typedef Dim<1>::SymTensor SymTensor;
  typedef Dim<1>::FacetedVolume FacetedVolume;

  const Scalar rin = 0.5*kernelExtent;

  // Copy the input positions to single list, and sort it.
  // Note our logic here relies on ghost nodes already being built, including parallel nodes.
  typedef pair<double, pair<unsigned, unsigned> > PointCoord;
  vector<PointCoord> coords;
  coords.reserve(numGens);
  for (unsigned nodeListi = 0; nodeListi != numNodeLists; ++nodeListi) {
    const unsigned n = position[nodeListi]->numElements();
    for (unsigned i = 0; i != n; ++i) {
      coords.push_back(make_pair(position(nodeListi, i).x(), make_pair(nodeListi, i)));
    }
  }
  sort(coords.begin(), coords.end(), ComparePairsByFirstElement<PointCoord>());

  // Now walk our sorted point and set the volumes and surface flags.
  surfacePoint = 0;
  const vector<NodeList<Dim<1> >*>& nodeListPtrs = position.nodeListPtrs();
  for (vector<PointCoord>::const_iterator itr = coords.begin();
       itr != coords.end();
       ++itr) {
    const unsigned nodeListi = itr->second.first;
    const unsigned i = itr->second.second;
    if (i < nodeListPtrs[nodeListi]->firstGhostNode()) {
      if (itr == coords.begin() or itr == coords.end()-1) {
        surfacePoint(nodeListi, i) = 1;
      } else {
        const unsigned nodeListj1 = (itr-1)->second.first,
                       nodeListj2 = (itr+1)->second.first,
                               j1 = (itr-1)->second.second,
                               j2 = (itr+1)->second.second;
        const Scalar Hi = H(nodeListi, i).xx(),
                    Hj1 = H(nodeListj1, j1).xx(),
                    Hj2 = H(nodeListj2, j2).xx();
        const Scalar xij1 = position(nodeListi, i).x() - position(nodeListj1, j1).x(),
                     xji2 = position(nodeListj2, j2).x() - position(nodeListi, i).x();
        CHECK(xij1 >= 0.0 and xji2 >= 0.0);
        const Scalar etamin = min(Hi, min(Hj1, Hj2))*min(xij1, xji2);
        if (etamin < rin) {
          vol(nodeListi, i) = 0.5*(xij1 + xji2);
        } else {
          surfacePoint(nodeListi, i) = 1;
        }
      }
    }
  }
}

}
}
