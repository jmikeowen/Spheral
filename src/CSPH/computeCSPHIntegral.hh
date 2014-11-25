//---------------------------------Spheral++------------------------------------
// Compute the CSPH corrections.
//------------------------------------------------------------------------------
#ifndef __Spheral__computeCSPHIntegral__
#define __Spheral__computeCSPHIntegral__

namespace Spheral {

  // Forward declarations.
  namespace NeighborSpace {
    template<typename Dimension> class ConnectivityMap;
  }
  namespace KernelSpace {
    template<typename Dimension> class TableKernel;
  }
  namespace FieldSpace {
    template<typename Dimension, typename DataType> class FieldList;
  }

  namespace CSPHSpace {

    template<typename Dimension>
    std::pair<typename Dimension::Vector,typename Dimension::Vector>
    computeCSPHIntegral(const NeighborSpace::ConnectivityMap<Dimension>& connectivityMap,
                           const KernelSpace::TableKernel<Dimension>& W,
                           const FieldSpace::FieldList<Dimension, typename Dimension::Scalar>& weight,
                           const FieldSpace::FieldList<Dimension, typename Dimension::Vector>& position,
                           const FieldSpace::FieldList<Dimension, typename Dimension::SymTensor>& H,
                           size_t nodeListi, const int i, size_t nodeListj, const int j, int mydim, const int order,
                           typename Dimension::Vector rmin, typename Dimension::Vector rmax);
  }
}

#endif
