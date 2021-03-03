//---------------------------------Spheral++----------------------------------//
// DamageGradientNodeCoupling
//
// A functor class encapsulating how we couple solid nodes in the presence of
// multiple materials and damage.
//
// This one attempts to mock up the shielding effect of ThreePointDamagedNodeCoupling
// by using local damage gradient to estimate when nodes are separated by
// regions of greater damage (or fractures).
//
// Created by JMO, Fri Jul 31 14:46:25 PDT 2015
//----------------------------------------------------------------------------//
#ifndef __Spheral_DamageGradientNodeCoupling__
#define __Spheral_DamageGradientNodeCoupling__

#include "DataBase/State.hh"
#include "Physics/Physics.hh"
#include "Kernel/TableKernel.hh"
#include "Utilities/NodeCoupling.hh"
#include "Utilities/DBC.hh"

namespace Spheral {

template<typename Dimension>
class DamageGradientNodeCoupling: public NodeCoupling {
public:
  using Scalar = typename Dimension::Scalar;
  using Vector = typename Dimension::Vector;
  using Tensor = typename Dimension::Tensor ;
  using SymTensor = typename Dimension::SymTensor;
  using ConstBoundaryIterator = typename Physics<Dimension>::ConstBoundaryIterator;

  // Constructor.
  DamageGradientNodeCoupling(const State<Dimension>& state,
                             const TableKernel<Dimension>& W,
                             ConstBoundaryIterator boundaryBegin,
                             ConstBoundaryIterator boundaryEnd,
                             NodePairList& pairs);
  virtual ~DamageGradientNodeCoupling() {}

private:
  // Forbidden methods.
  DamageGradientNodeCoupling();
};

}

#endif
