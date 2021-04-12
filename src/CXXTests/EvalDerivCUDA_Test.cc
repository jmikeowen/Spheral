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
#include "LvArray/ChaiBuffer.hpp"

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
using Array1D = LvArray::Array< T, 1, camp::idx_seq<0>, std::ptrdiff_t, LvArray::ChaiBuffer >;

template<typename T>
using Array1DView = LvArray::ArrayView< T, 1, 0, std::ptrdiff_t, LvArray::ChaiBuffer >;

template<typename T>
struct FieldAccessor {
  using array_type = Array1D<T>;
  using view_type = Array1DView<T>;

  FieldAccessor(const array_type& arr) : array_parent(arr), view(arr) {}

  size_t size() const { return view.size(); }

  template<typename IDX_TYPE>
  RAJA_HOST_DEVICE T& operator[](const IDX_TYPE idx) const { return view[idx]; }

  void move( const LvArray::MemorySpace space ) { array_parent.move(space); }
   
private:
  const array_type& array_parent;
  const view_type& view;
};


int main() {

  constexpr int N = 50000;

  // Create Basic NodeList
  using Dim = Spheral::Dim<3>;
  Spheral::NodeList<Dim> node_list("example_node_list", N, 0);
  auto n_pos = node_list.positions();

  FieldAccessor< Spheral::GeomVector<3> > field_view( n_pos.mDataArray );

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, field_view.size()), [=](unsigned int kk) {
      field_view[kk][0]++;
      field_view[kk][1]++;
      field_view[kk][2]++;
  });

  field_view.move( LvArray::MemorySpace::GPU );

  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, field_view.size()), [=] RAJA_HOST_DEVICE (int kk) {
      field_view[kk][0]++;
      field_view[kk][1]++;
      field_view[kk][2]++;
  });

  field_view.move( LvArray::MemorySpace::CPU );

  bool correctness = true;
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, field_view.size()), [&] (int kk) {
      if (n_pos[kk] != Spheral::GeomVector<3>(2,2,2)) correctness = false;
  });

  if (correctness)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";

  return EXIT_SUCCESS;
}
