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
#include <typeinfo>

#include "ArtificialViscosity/MonaghanGingoldViscosity.hh"

#include "DataBase/DataBase.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"

#include "Geometry/Dimension.hh"

#include "Hydro/HydroFieldNames.hh"

#include "Integrator/CheapSynchronousRK2.hh"

#include "NodeList/NodeList.hh"

#include "SPH/SPHHydroBase.hh"

#include "Utilities/DataTypeTraits.hh"
#include "Utilities/SpheralTimers.cc"

#include "LvArray/Array.hpp"
#include "LvArray/MallocBuffer.hpp"

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


namespace Spheral {
namespace expt {

//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
evaluateDerivatives(const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivatives) {

  typedef typename Dimension::Vector Vector;

  TIME_SPHevalDerivs.start();
  TIME_SPHevalDerivs_initial.start();

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

  TIME_SPHevalDerivs_initial.stop();

  // Walk all the interacting pairs.
  TIME_SPHevalDerivs_pairs.start();
  std::cout << position.numNodes() << "\n";
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

  // Finish up the derivatives for each point.
  TIME_SPHevalDerivs.stop();
}

} //  namespace expt
} //  namespace Spheral

template<typename T>
using Array1D = LvArray::Array< T, 1, camp::idx_seq<0>, std::ptrdiff_t, LvArray::MallocBuffer >;

int main() {

  // Create Basic NodeList
  //using Dim = Spheral::Dim<1>;
  //Spheral::NodeList<Dim> node_list("example_node_list", 10000, 0);
  //auto n_pos = node_list.positions();

  Array1D< Spheral::GeomVector<3> > array(5);

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, array.size()), [=](unsigned int kk) {
      printf("%f, %f, %f\n", array[kk][0], array[kk][1], array[kk][2] );
  });

  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, array.size()), [=] RAJA_HOST_DEVICE (int kk) {
      Spheral::GeomVector<3> g_vec(kk,kk,kk);
      printf("%f, %f, %f\n", g_vec[0], g_vec[1], g_vec[2] );
  });

  return EXIT_SUCCESS;
}
