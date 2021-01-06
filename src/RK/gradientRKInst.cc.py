text = """
//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "RK/gradientRK.cc"
#include "Geometry/Dimension.hh"

namespace Spheral {
"""

for Value in ("Scalar", "Vector"):
    text += """
template 
FieldList<Dim<%%(ndim)s>, MathTraits<Dim<%%(ndim)s>, Dim<%%(ndim)s>::%(Value)s>::GradientType> 
gradientRK<Dim<%%(ndim)s>, Dim<%%(ndim)s>::%(Value)s>(const FieldList<Dim<%%(ndim)s>, Dim<%%(ndim)s>::%(Value)s>& fieldList,
                                                      const FieldList<Dim<%%(ndim)s>, Dim<%%(ndim)s>::Vector>& position,
                                                      const FieldList<Dim<%%(ndim)s>, Dim<%%(ndim)s>::Scalar>& weight,
                                                      const FieldList<Dim<%%(ndim)s>, Dim<%%(ndim)s>::SymTensor>& H,
                                                      const ConnectivityMap<Dim<%%(ndim)s>>& connectivityMap,
                                                      const ReproducingKernel<Dim<%%(ndim)s>>& WR,
                                                      const FieldList<Dim<%%(ndim)s>, RKCoefficients<Dim<%%(ndim)s>>>& corrections,
                                                      const NodeCoupling& nodeCoupling);

""" % {"Value" : Value}

    text += """
template 
FieldList<Dim<%%(ndim)s>, std::vector<MathTraits<Dim<%%(ndim)s>, Dim<%%(ndim)s>::%(Value)s>::GradientType>> 
gradientRK<Dim<%%(ndim)s>, Dim<%%(ndim)s>::%(Value)s>(const FieldList<Dim<%%(ndim)s>, std::vector<Dim<%%(ndim)s>::%(Value)s>>& fieldList,
                                                      const FieldList<Dim<%%(ndim)s>, Dim<%%(ndim)s>::Vector>& position,
                                                      const FieldList<Dim<%%(ndim)s>, Dim<%%(ndim)s>::Scalar>& weight,
                                                      const FieldList<Dim<%%(ndim)s>, Dim<%%(ndim)s>::SymTensor>& H,
                                                      const ConnectivityMap<Dim<%%(ndim)s>>& connectivityMap,
                                                      const ReproducingKernel<Dim<%%(ndim)s>>& WR,
                                                      const FieldList<Dim<%%(ndim)s>, RKCoefficients<Dim<%%(ndim)s>>>& corrections,
                                                      const NodeCoupling& nodeCoupling);

""" % {"Value" : Value}
    
text += """
}
"""
