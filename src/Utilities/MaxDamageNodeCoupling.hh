//---------------------------------Spheral++----------------------------------//
// MaxDamageNodeCoupling
//
// A very simple form of pair-wise damage coupling -- just apply the maximum
// of the damage on the two points.
//
// Created by JMO, Tue Nov 10 10:37:54 PST 2020
//----------------------------------------------------------------------------//
#ifndef __Spheral_MaxDamageNodeCoupling__
#define __Spheral_MaxDamageNodeCoupling__

#include "Utilities/NodeCoupling.hh"

namespace Spheral {

template<typename Dimension>
class MaxDamageNodeCoupling: public NodeCoupling {
public:
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::SymTensor SymTensor;

  // Constructor.
  MaxDamageNodeCoupling(const FieldList<Dimension, SymTensor>& damage):
    NodeCoupling(),
    mDamage(damage) {}

  // The coupling operator.
  virtual double operator()(const unsigned nodeListi, const unsigned i,
                            const unsigned nodeListj, const unsigned j) const {
    return std::max(0.0, std::min(1.0, 1.0 - std::max(scalarDamage(nodeListi, i), scalarDamage(nodeListj, j))));
  }

private:
  const FieldList<Dimension, SymTensor>& mDamage;

  // Extract the effective scalar damage on a node.
  double scalarDamage(const unsigned nodeListi, const unsigned i) const {
    Scalar sDi = std::max(0.0, std::min(1.0, mDamage(nodeListi, i).eigenValues().maxElement()));
    if (sDi > 1.0 - 1.0e-3) sDi = 1.0;
    return sDi;
  }

  // Forbidden methods.
  MaxDamageNodeCoupling();
};

}

#endif
