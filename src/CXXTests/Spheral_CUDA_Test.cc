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

#include "Field/Field.hh"

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
  using EXEC_POL = RAJA::cuda_exec<256>;
  using REDUCE_POL = RAJA::cuda_reduce;
  #define OFFLOAD_SPACE LvArray::MemorySpace::GPU
#elif defined(RAJA_ENABLE_OPENMP)
  using EXEC_POL = RAJA::omp_for_exec;
  using REDUCE_POL = RAJA::omp_reduce;
  #define OFFLOAD_SPACE LvArray::MemorySpace::CPU
#else
  using EXEC_POL = RAJA::seq_exec;
  using REDUCE_POL = RAJA::seq_reduce;
  #define OFFLOAD_SPACE LvArray::MemorySpace::CPU
#endif
  #define HOST_SPACE LvArray::MemorySpace::CPU
  using HOST_POL = RAJA::seq_exec;


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
  RAJA::forall<EXEC_POL>(array_npairs, [&](unsigned int kk) {
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

void setupNodePairList(Spheral::NodePairList& npl, const size_t sz) {
  for (size_t i = 0; i < sz; i++) {
    npl.push_back(Spheral::NodePairIdxType(0,0,0,0));
  }
}

namespace Spheral{
  namespace detail{

#define DEVICE_ACCESSOR(VAL) DeviceAccessor<decltype(VAL)>(VAL)

    template<typename T>
    class DeviceAccessor {

      using value_type = typename T::ValueType;
      using array_type = typename T::ContainerType;
      using view_type  = typename T::ContainerTypeView;

    public:
      DeviceAccessor(const T& val) : array_parent(val.mDataArray), view(val.mDataArray) {}

      unsigned size() const { return view.size(); }

      template<typename IDX_TYPE>
      RAJA_HOST_DEVICE value_type& operator[](const IDX_TYPE idx) const { return view[idx]; }

      void move( const LvArray::MemorySpace space ) {
      #if defined(RAJA_ENABLE_CUDA)
        array_parent.move(space);
      #else
        RAJA_UNUSED_VAR(space);
      #endif
      }
       
    private:
      const array_type& array_parent;
      const view_type& view;
    };

  }
}


int main() {

  constexpr int N = 50000;

  // Create Basic NodeList
  using Dim = Spheral::Dim<3>;
  Spheral::NodeList<Dim> node_list("example_node_list", N, 0);

  auto n_pos = node_list.positions();
  Spheral::NodePairList npl;
  setupNodePairList(npl, N);

  auto field_view = Spheral::detail::DEVICE_ACCESSOR(n_pos);
  auto npl_view = Spheral::detail::DEVICE_ACCESSOR(npl);

  RAJA::RangeSegment range(0, field_view.size());
  RAJA::forall<HOST_POL>(range,
    [=](unsigned int kk) {
      Spheral::NodePairIdxType np(kk,0,kk,0);
      npl_view[kk] = np;

      field_view[kk][0]++;
      field_view[kk][1]++;
      field_view[kk][2]++;
  });

  field_view.move(OFFLOAD_SPACE);
  npl_view.move(OFFLOAD_SPACE);

  RAJA::forall<EXEC_POL>(range, 
    [=] RAJA_HOST_DEVICE (int kk) {
      Spheral::NodePairIdxType np(kk,kk,kk,kk);
      npl_view[kk] = np;

      field_view[kk][0]++;
      field_view[kk][1]++;
      field_view[kk][2]++;
  });

  field_view.move(HOST_SPACE);
  npl_view.move(HOST_SPACE);

  bool correctness = true;
  RAJA::forall<HOST_POL>(range,
    [&] (int kk) {
      if (n_pos[kk] != Spheral::GeomVector<3>(2,2,2)) correctness = false;
      if (npl[kk] != Spheral::NodePairIdxType(kk,kk,kk,kk)) correctness = false;
  });

  if (correctness)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";

  return EXIT_SUCCESS;
}
