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


int main() {
  using Dim = Spheral::Dim<1>;

  // Create Basic NodeList
  Spheral::NodeList<Dim> node_list("example_node_list", 10, 0);

  // Pass NodeList to Database.
  Spheral::DataBase<Dim> db;
  db.appendNodeList(node_list);

  // Build up a Physics Package
  //Spheral::MonaghanGingoldViscosity<Dim> Q(1,1, false, false);
  //Spheral::GenericHydro<Dim> basic_physics(Q, 1, false);

  // Make an Integrator Object.
  //std::vector<Spheral::Physics<Dim>*> packages{&basic_physics};
  std::vector<Spheral::Physics<Dim>*> packages{};
  Spheral::CheapSynchronousRK2<Dim> integrator(db, packages);

  // Generate a State Objects from Database.
  auto physics_packages = integrator.physicsPackages();
  Spheral::State<Dim> state(db, physics_packages);
  Spheral::StateDerivatives<Dim> state_derivs(db, physics_packages);

  //Spheral::expt::evaluateDerivatives<Dim>(db, state, state_derivs);

  auto n_pos = node_list.positions();
  RAJA::TypedRangeSegment<unsigned int> n_nodes(0, node_list.numNodes());

  RAJA::forall<PAIR_EXEC_POL>(n_nodes, [=](unsigned int kk) {
      const auto& x = n_pos[kk];
      printf("%f\n", x[0]);
  });

  //auto allpos = db.globalPosition();
  //for (size_t i = 0; i < allpos.numNodes(); i++) {
  //  std::cout << allpos(0, i) << "\n";
  //}

  return EXIT_SUCCESS;
}
