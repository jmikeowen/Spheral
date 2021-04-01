// ------------------------------------------------------------------------------------------
// Test Executable to populate a FieldList with basic data, manipulate it and checkit again.
// ------------------------------------------------------------------------------------------

// Create a set of nodes with a given position.
// Create a position field for given set of nodes.
// Register position field with a state.
//
// Create randomized "fake" connectivity Map
// Generate pairlist from connectivity map.
//
// Pass State to fake Eval Derivs
// Move data to GPU
// Update "Dpos" of nodes based on pairs position? 
//   Maybe increment by mid distance of the pair?
// Move data back to Host.
//
// For each node (Maybe on GPU???)
//   Get Dpos, divide by numNeighbors.
//   Increment position state value by Dpos 
//
// Check correctness.

#include <iostream>
#include "DataBase/DataBase.hh"
#include "Utilities/SpheralTimers.cc"
#include "Hydro/HydroFieldNames.hh"
#include "Utilities/DataTypeTraits.hh"

#include "NodeList/NodeList.hh"
#include "Geometry/Dimension.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
evaluateDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar /*dt*/,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivatives) {

  typedef typename Dimension::Vector Vector;

  TIME_SPHevalDerivs.start();
  TIME_SPHevalDerivs_initial.start();

#if defined(RAJA_ENABLE_CUDA)
  using PAIR_EXEC_POL = RAJA::cuda_exec<256>;
  using PAIR_REDUCE_POL = RAJA::cuda_reduce;
  using NODE_INNER_EXEC_POL = RAJA::seq_exec;
#elif defined(RAJA_ENABLE_OPENMP)
  using PAIR_EXEC_POL = RAJA::omp_for_exec;
  using PAIR_REDUCE_POL = RAJA::omp_reduce;
  using NODE_INNER_EXEC_POL = RAJA::omp_for_exec;
#else
  using PAIR_EXEC_POL = RAJA::seq_exec;
  using PAIR_REDUCE_POL = RAJA::seq_reduce;
  using NODE_INNER_EXEC_POL = RAJA::seq_exec;
#endif
  using NODE_OUTER_EXEC_POL = RAJA::seq_exec;


  // A few useful constants we'll use in the following loop.
  const double tiny = 1.0e-30;

  // The connectivity.
  const auto& connectivityMap = dataBase.connectivityMap();
  const auto& nodeLists = connectivityMap.nodeLists();
  const auto numNodeLists = nodeLists.size();

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();

  // Get the state and derivative FieldLists.
  // State FieldLists.
  auto position = state.fields(HydroFieldNames::position,
                               Vector::zero);
  auto velocity = state.fields(HydroFieldNames::velocity,
                               Vector::zero);

  //auto  DxDt = derivatives.fields(IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, Vector::zero);
  //auto  DvDt = derivatives.fields(HydroFieldNames::hydroAcceleration, Vector::zero);
  //auto  DvDx = derivatives.fields(HydroFieldNames::velocityGradient, Tensor::zero);

  //auto DvDt_reducer = DvDt.getReduceSum(PAIR_REDUCE_POL());
  //auto DvDx_reducer = DvDx.getReduceSum(PAIR_REDUCE_POL());

  // The scale for the tensile correction.
  //const auto& nodeList = mass[0]->nodeList();

  TIME_SPHevalDerivs_initial.stop();

  // Walk all the interacting pairs.
  TIME_SPHevalDerivs_pairs.start();
  RAJA::TypedRangeSegment<unsigned int> array_npairs(0, npairs);
  RAJA::forall<PAIR_EXEC_POL>(array_npairs, [&](unsigned int kk) {
      //
    // Thread private scratch variables
    int i, j, nodeListi, nodeListj;

    i = pairs[kk].i_node;
    j = pairs[kk].j_node;
    nodeListi = pairs[kk].i_list;
    nodeListj = pairs[kk].j_list;

  });
  TIME_SPHevalDerivs_pairs.stop();

  //DvDt.getReduction(DvDt_reducer);
  //DvDx.getReduction(DvDx_reducer);

  // Finish up the derivatives for each point.
  TIME_SPHevalDerivs.stop();
}

}

int main() {

  std::cout << "Hello World\n";

  using Dim = Spheral::Dim<1>;

  Spheral::NodeList<Dim> node_list("example_node_list", 10, 0);

  return EXIT_SUCCESS;
}
