//---------------------------------Spheral++------------------------------------
// Compute the RK second-derivative
//------------------------------------------------------------------------------
#ifndef __Spheral__hessianRK__
#define __Spheral__hessianRK__

#include "Geometry/MathTraits.hh"
#include "SPH/NodeCoupling.hh"
#include "RK/RKCorrectionParams.hh"

namespace Spheral {

// Forward declarations.
template<typename Dimension> class ConnectivityMap;
template<typename Dimension> class TableKernel;
template<typename Dimension, typename DataType> class FieldList;

template<typename Dimension, typename DataType>
FieldList<Dimension, typename MathTraits<Dimension, DataType>::HessianType>
hessianRK(const FieldList<Dimension, DataType>& fieldList,
          const FieldList<Dimension, typename Dimension::Vector>& position,
          const FieldList<Dimension, typename Dimension::Scalar>& weight,
          const FieldList<Dimension, typename Dimension::SymTensor>& H,
          const ConnectivityMap<Dimension>& connectivityMap,
          const TableKernel<Dimension>& W,
          const RKOrder correctionOrder,
          const FieldList<Dimension, std::vector<double>>& corrections,
          const NodeCoupling& nodeCoupling = NodeCoupling());

}

#endif
