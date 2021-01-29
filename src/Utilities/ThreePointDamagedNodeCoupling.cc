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
#include "Utilities/ThreePointDamagedNodeCoupling.hh"
#include "Utilities/pointDistances.hh"
#include "Utilities/DBC.hh"
#include "Utilities/Timer.hh"

#include <vector>

using std::vector;

// Declare timers
extern Timer TIME_Damage;
extern Timer TIME_ThreePointCoupling;
extern Timer TIME_ThreePointCoupling_initial;
extern Timer TIME_ThreePointCoupling_pairs;

namespace Spheral {

//------------------------------------------------------------------------------
// The function we use to damage the pair coupling
//------------------------------------------------------------------------------
template<typename Scalar, typename SymTensor, typename Vector, typename Kernel>
inline
double pairCoupling(const Vector& xk,
                    const SymTensor& Hk,
                    const SymTensor& Dk,
                    const SymTensor& Di,
                    const SymTensor& Dj,
                    const Vector& xhatji,
                    const Vector& b,
                    const Kernel& W,
                    const Scalar& W0) {
  const auto dk = (Dk*xhatji).magnitude();
  const auto di = (Di*xhatji).magnitude();
  const auto dj = (Dj*xhatji).magnitude();
  CHECK(dk >= 0.0 and dk <= 1.0);
  CHECK(di >= 0.0 and di <= 1.0);
  CHECK(dj >= 0.0 and dj <= 1.0);
  if (std::max(di, dj) < 1.0e-3) return 1.0;
  return std::max(0.0, std::min(1.0, 1.0 - dk * W.kernelValue((Hk*(b - xk)).magnitude(), 1.0)/W0));
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template<typename Dimension>
ThreePointDamagedNodeCoupling<Dimension>::
ThreePointDamagedNodeCoupling(const FieldList<Dimension, Vector>& position,
                              const FieldList<Dimension, SymTensor>& H,
                              const FieldList<Dimension, SymTensor>& damage,
                              const TableKernel<Dimension>& W,
                              const ConnectivityMap<Dimension>& connectivity,
                              NodePairList& pairs) {

  TIME_Damage.start();
  TIME_ThreePointCoupling.start();
  const auto W0 = W.kernelValue(0.0, 1.0);
  const auto etamax2 = FastMath::square(W.kernelExtent());
  const auto npairs = pairs.size();
  const auto numNodeLists = position.numFields();
  const auto Dthreshold = 1.0e-3;
  const auto useIntersectConnectivity = connectivity.buildIntersectionConnectivity();
  // const auto Dfull = 0.999;

  // For each interacting pair we need to compute the effective damage shielding, expressed
  // as the f_couple parameter in the NodePairIdxType.
  // Everyone starts out fully coupled.
  TIME_ThreePointCoupling_initial.start();
#pragma omp parallel for
  for (auto i = 0u; i < npairs; ++i) {
    pairs[i].f_couple = 1.0;
  }

  // Flag all points that interact with a damaged point.
  auto workToBeDone = false;
  FieldList<Dimension, unsigned> flags(FieldStorageType::CopyFields);
  for (auto il = 0u; il < numNodeLists; ++il) {
    flags.appendNewField("damage interaction", damage[il]->nodeList(), 0u);
    const auto ni = damage[il]->numElements();
#pragma omp parallel for
    for (auto i = 0u; i < ni; ++i) {
      if (damage(il,i).Trace() > Dthreshold) {
        const auto& connectivity_i = connectivity.connectivityForNode(il,i);
        for (auto jl = 0u; jl < numNodeLists; ++jl) {
          for (const auto j: connectivity_i[jl]) {
            flags(jl,j) = 1u;
          }
        }
        flags(il,i) = 1u;
#pragma omp atomic write
        workToBeDone = true;
      }
    }
  }
  // Parallel note: at this point workToBeDone is rank dependent, so some ranks will enter the following
  // block and some not.
  TIME_ThreePointCoupling_initial.stop();

  // Now apply damage to pair interactions.
  if (workToBeDone) {

    if (useIntersectConnectivity) {
      // This branch uses the fact the ConnectivityMap has already computed the intersection
      // of each pair.
      TIME_ThreePointCoupling_pairs.start();
      Vector b;
#pragma omp parallel for private(b)
      for (auto kk = 0u; kk < npairs; ++kk) {
        auto& pair = pairs[kk];
        if (flags(pair.i_list, pair.i_node) == 1u or
            flags(pair.j_list, pair.j_node) == 1u) {
          auto& fij = pair.f_couple;
          const auto& xi = position(pair.i_list, pair.i_node);
          const auto& xj = position(pair.j_list, pair.j_node);
          const auto& Di = damage(pair.i_list, pair.i_node);
          const auto& Dj = damage(pair.j_list, pair.j_node);
          const auto  xji = xj - xi;
          const auto  xhatji = xji.unitVector();
          // const bool barf = pair == NodePairIdxType(48, 0, 49, 0);
          // if (barf) std::cerr << "  3pt firing on " << pair << " : " << connectivity.numNeighborsForNode(pair.i_list, pair.i_node) << " " << connectivity.numNeighborsForNode(pair.j_list, pair.j_node) << std::endl;

          // Find the common neighbors for this pair.
          const auto& intersection_list = connectivity.intersectionConnectivity(pair);
          auto nodeListk = 0u;
          while (nodeListk < numNodeLists and fij > 1e-10) {
            for (const auto k: intersection_list[nodeListk]) {

              // State for node k
              const auto& xk = position(nodeListk, k);
              const auto& Hk = H(nodeListk, k);
              const auto& Dk = damage(nodeListk, k);

              // We only proceed if the closest point to k on (i,j) is bounded by (i,j)
              if (Dk.Trace() > Dthreshold and closestPointOnSegment(xk, xi, xj, b)) {
                fij *= pairCoupling(xk, Hk, Dk, Di, Dj, xhatji, b, W, W0);
              }
              // if (barf) std::cerr << "  --> " << k << " " << Dk << " " << pairCoupling(xk, Hk, Dk, Di, Dj, xhatji, b, W, W0) << " " << fij << std::endl;
              if (fij < 1.0e-10) break;
            }
            ++nodeListk;
          }
          CHECK(fij >= 0.0 and fij <= 1.0);
        }
      }
      TIME_ThreePointCoupling_pairs.stop();

    } else {
      // This branch computes the same f_couple as above, but doesn't rely on the explicit pre-calcuation of intersection
      // sets.  Instead we loop over each point and see which pairs it interacts with.
      for (auto kl = 0u; kl < numNodeLists; ++kl) {
        const auto nk = position[kl]->numElements();
#pragma omp parallel
        {
          Vector b;
#pragma omp for
          for (auto k = 0u; k < nk; ++k) {
            const auto& Dk = damage(kl, k);
            if (Dk.Trace() > Dthreshold) {
              // std::cout << "3pt firing on (" << kl << " " << k << ") " << Dk.Trace() << " : " << connectivity.numNeighborsForNode(kl, k) << std::endl;

              // This point has damage, so damage the pair interactions for all pairs that k is in
              // the intersection set of.
              const auto& xk = position(kl, k);
              const auto& Hk = H(kl, k);
              const auto& connectivity_k = connectivity.connectivityForNode(kl, k);
              CHECK(connectivity_k.size() == numNodeLists);
              for (auto il = 0u; il  < numNodeLists; ++il) {
                const auto ni = connectivity_k[il].size();
                for (auto ii = 0u; ii < ni; ++ii) {
                  const auto  i = connectivity_k[il][ii];
                  const auto& xi = position(il, i);
                  const auto& Hi = H(il, i);
                  const auto& Di = damage(il, i);
                  for (auto jl = il; jl < numNodeLists; ++jl) {
                    const auto nj = connectivity_k[jl].size();
                    for (auto jj = (jl == il ? ii + 1u : 0u); jj < nj; ++jj) {
                      const auto  j = connectivity_k[jl][jj];
                      const auto& xj = position(jl, j);
                      const auto& Hj = H(jl, j);
                      const auto& Dj = damage(jl, j);
                      const auto  xji = xj - xi;
                      if (std::min((Hi*xji).magnitude2(), (Hj*xji).magnitude2()) < etamax2) {
                        // k is in the intersection of (i,j) connectivity, but is it geometrically between (i,j)?
                        if (closestPointOnSegment(xk, xi, xj, b)) {
                          // Yep!
                          const auto xhatji = xji.unitVector();
                          const auto pair = std::min(NodePairIdxType(i, il, j, jl), NodePairIdxType(j, jl, i, il));
                          const auto itr = std::lower_bound(pairs.begin(), pairs.end(), pair);
                          if (itr < pairs.end() and *itr == pair) {
#pragma omp atomic
                            itr->f_couple *= pairCoupling(xk, Hk, Dk, Di, Dj, xhatji, b, W, W0);
                            // if (pair == NodePairIdxType(48, 0, 49, 0)) std::cerr << "  1-> " << i << " " << j << " " << k << " " << Dk << " " << pairCoupling(xk, Hk, Dk, Di, Dj, xhatji, b, W, W0) << " " << itr->f_couple << std::endl;
                          }
                        }
                      }
                    }
                  }
                  // Damage (k,i)
                  const NodePairIdxType pair(k, kl, i, il);
                  const auto itr = std::find(pairs.begin(), pairs.end(), pair);
                  if (itr != pairs.end() and *itr == pair) {
                    const auto xhatik = (xk - xi).unitVector();
#pragma omp atomic
                    itr->f_couple *= pairCoupling(Vector::zero, Hk, Dk, Di, Dk, xhatik, Vector::zero, W, W0);
                    // if (pair == NodePairIdxType(48, 0, 49, 0)) std::cerr << "  3-> " << i << " " << k << " " << k << " " << Dk << " " << pairCoupling(xk, Hk, Dk, Di, Dk, xhatik, Vector::zero, W, W0) << " " << itr->f_couple << std::endl;
                  }                   
                }
              }
            }
          }
        }  // omp parallel
      }
    }
  }
  
  TIME_ThreePointCoupling.stop();
  TIME_Damage.stop();
}

}
