//---------------------------------Spheral++------------------------------------
// Compute the RK gradient.
//------------------------------------------------------------------------------
#ifndef __Spheral__gradientRK__
#define __Spheral__gradientRK__

#include "Geometry/MathTraits.hh"
#include "Utilities/NodeCoupling.hh"
#include "RK/RKCorrectionParams.hh"
#include "RK/ReproducingKernel.hh"

namespace Spheral {

// Forward declarations.
template<typename Dimension> class ConnectivityMap;
template<typename Dimension, typename DataType> class FieldList;

template<typename Dimension, typename DataType>
FieldList<Dimension, typename MathTraits<Dimension, DataType>::GradientType>
gradientRK(const FieldList<Dimension, DataType>& fieldList,
           const FieldList<Dimension, typename Dimension::Vector>& position,
           const FieldList<Dimension, typename Dimension::Scalar>& weight,
           const FieldList<Dimension, typename Dimension::SymTensor>& H,
           const ConnectivityMap<Dimension>& connectivityMap,
           const ReproducingKernel<Dimension>& WR,
           const FieldList<Dimension, RKCoefficients<Dimension>>& corrections,
           const NodeCoupling& nodeCoupling = NodeCoupling());

template<typename Dimension, typename DataType>
FieldList<Dimension, std::vector<typename MathTraits<Dimension, DataType>::GradientType>>
gradientRK(const FieldList<Dimension, std::vector<DataType>>& fieldList,
           const FieldList<Dimension, typename Dimension::Vector>& position,
           const FieldList<Dimension, typename Dimension::Scalar>& weight,
           const FieldList<Dimension, typename Dimension::SymTensor>& H,
           const ConnectivityMap<Dimension>& connectivityMap,
           const ReproducingKernel<Dimension>& WR,
           const FieldList<Dimension, RKCoefficients<Dimension>>& corrections,
           const NodeCoupling& nodeCoupling = NodeCoupling());
}

#endif
