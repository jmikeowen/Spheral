//---------------------------------Spheral++----------------------------------//
// ThreePointDamagedNodeCoupling
//
// A functor class encapsulating how we couple solid nodes in the presence of
// multiple materials and damage.
// 
// This for uses the "three point" formalism, which allows damaged points to
// cut communication between pairs that talk across them.
//
// Unlike other NodeCouping objects, ThreePointDamagedNodeCoupling directly
// sets the pairwise f_couple in the NodePairList, so all work is done in the
// constructor.
//
// Created by JMO, Fri Jan  1 15:18:38 PST 2021
//----------------------------------------------------------------------------//
#ifndef __Spheral_ThreePointDamagedNodeCoupling__
#define __Spheral_ThreePointDamagedNodeCoupling__

#include "Utilities/NodeCoupling.hh"
#include "DataBase/State.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Kernel/TableKernel.hh"

namespace Spheral {

template<typename Dimension>
class ThreePointDamagedNodeCoupling: public NodeCoupling {
public:
  using Scalar = typename Dimension::Scalar;
  using Vector = typename Dimension::Vector;
  using Tensor = typename Dimension::Tensor ;
  using SymTensor = typename Dimension::SymTensor;

  // Constructor.
  ThreePointDamagedNodeCoupling(const State<Dimension>& state,
                                const TableKernel<Dimension>& W,
                                NodePairList& pairs);

private:
  // Forbidden methods.
  ThreePointDamagedNodeCoupling();
};

}

#endif
