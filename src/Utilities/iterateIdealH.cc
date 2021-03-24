//------------------------------------------------------------------------------
// Iterate the ideal H algorithm to converge on a new H field.
// This routine replaces the H field in place.
//------------------------------------------------------------------------------
#include "iterateIdealH.hh"
#include "Field/FieldList.hh"
#include "NodeList/SmoothingScaleBase.hh"
#include "Utilities/allReduce.hh"
#include "Distributed/Communicator.hh"
#include "Geometry/GeometryRegistrar.hh"

#include <ctime>
using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::abs;

namespace Spheral {

template<typename Dimension>
void
iterateIdealH(DataBase<Dimension>& dataBase,
              const vector<Boundary<Dimension>*>& boundaries,
              const TableKernel<Dimension>& W,
              const SmoothingScaleBase<Dimension>& smoothingScaleMethod,
              const int maxIterations,
              const double tolerance,
              const double nPerhForIteration,
              const bool sphericalStart,
              const bool fixDeterminant) {

  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::SymTensor SymTensor;

  // Start the timing.
  const auto t0 = clock();

  // Extract the state we care about.
  const auto pos = dataBase.fluidPosition();
  auto m = dataBase.fluidMass();
  auto rho = dataBase.fluidMassDensity();
  auto H = dataBase.fluidHfield();

  // If we're using the fixDeterminant, take a snapshot of the input H determinants.
  auto Hdet0 = dataBase.newFluidFieldList(double());
  if (fixDeterminant) {
    for (auto itr = dataBase.internalNodeBegin();
         itr != dataBase.internalNodeEnd();
         ++itr) Hdet0(itr) = H(itr).Determinant();
  }

  // Store the input nperh for each NodeList.
  // If we're rescaling the nodes per h for our work, make a cut at it.
  vector<double> nperh0;
  for (auto nodeListItr = dataBase.fluidNodeListBegin();
       nodeListItr != dataBase.fluidNodeListEnd(); 
       ++nodeListItr) {
    const auto nperh = (*nodeListItr)->nodesPerSmoothingScale();
    nperh0.push_back(nperh);
    if (distinctlyGreaterThan(nPerhForIteration, 0.0)) {
      auto& Hfield = **(H.fieldForNodeList(**nodeListItr));
      Hfield *= Dimension::rootnu(nperh/nPerhForIteration);
      (*nodeListItr)->nodesPerSmoothingScale(nPerhForIteration);
    }
  }
  CHECK(nperh0.size() == dataBase.numFluidNodeLists());

  // If we are both fixing the H determinant and rescaling the nodes per h,
  // we need a snapshot of the rescaled H determinant as well.
  auto Hdet1 = dataBase.newFluidFieldList(double());
  if (fixDeterminant) {
    for (auto itr = dataBase.internalNodeBegin();
         itr != dataBase.internalNodeEnd();
         ++itr) Hdet1(itr) = H(itr).Determinant();
  }

  // If requested, start by making all the H's round (volume preserving).
  if (sphericalStart) {
    for (auto itr = dataBase.internalNodeBegin();
         itr != dataBase.internalNodeEnd();
         ++itr) {
      const auto Hdeti = H(itr).Determinant();
      const auto Hi = Dimension::rootnu(Hdeti) * SymTensor::one;
      H(itr) = Hi;
    }
  }

  // Build a list of flags to indicate which nodes have been completed.
  auto flagNodeDone = dataBase.newFluidFieldList(0, "node completed");

  // Iterate until we either hit the max iterations or the H's achieve convergence.
  const auto numNodeLists = dataBase.numFluidNodeLists();
  auto maxDeltaH = 2.0*tolerance;
  auto itr = 0;
  while (itr < maxIterations and maxDeltaH > tolerance) {
    ++itr;
    maxDeltaH = 0.0;
    // flagNodeDone = 0;

    // Remove any old ghost node information from the NodeLists.
    for (auto k = 0u; k < numNodeLists; ++k) {
      auto nodeListPtr = *(dataBase.fluidNodeListBegin() + k);
      nodeListPtr->numGhostNodes(0);
      nodeListPtr->neighbor().updateNodes();
    }

    // Enforce boundary conditions.
    for (auto k = 0u; k < boundaries.size(); ++k) {
      auto boundaryPtr = *(boundaries.begin() + k);
      boundaryPtr->setAllGhostNodes(dataBase);
      boundaryPtr->applyFieldListGhostBoundary(m);
      boundaryPtr->applyFieldListGhostBoundary(rho);
      boundaryPtr->finalizeGhostBoundary();
      for (auto nodeListItr = dataBase.fluidNodeListBegin();
           nodeListItr != dataBase.fluidNodeListEnd(); 
           ++nodeListItr) {
        (*nodeListItr)->neighbor().updateNodes();
      }
    }

    // Prepare a FieldList to hold the new H.
    FieldList<Dimension, SymTensor> H1(H);
    H1.copyFields();
    auto zerothMoment = dataBase.newFluidFieldList(0.0, "zerothMoment");
    auto secondMoment = dataBase.newFluidFieldList(SymTensor::zero, "secondMoment");

    // Get the new connectivity.
    dataBase.updateConnectivityMap(false, false, false);
    const auto& connectivityMap = dataBase.connectivityMap();
    const auto& pairs = connectivityMap.nodePairList();
    const auto  npairs = pairs.size();

    // Walk the pairs.
#pragma omp parallel
    {
      typename SpheralThreads<Dimension>::FieldListStack threadStack;
      auto zerothMoment_thread = zerothMoment.threadCopy(threadStack);
      auto secondMoment_thread = secondMoment.threadCopy(threadStack);

      int i, j, nodeListi, nodeListj;
      Scalar ri, rj, mRZi, mRZj, Wi, gWi, Wj, gWj;
      Vector xij, etai, etaj, gradWi, gradWj;
      SymTensor thpt;

#pragma omp for
      for (auto k = 0u; k < npairs; ++k) {
        i = pairs[k].i_node;
        j = pairs[k].j_node;
        nodeListi = pairs[k].i_list;
        nodeListj = pairs[k].j_list;

        // Anything to do?
        if (flagNodeDone(nodeListi, i) == 0 or flagNodeDone(nodeListj, j) == 0) {
          const auto& posi = pos(nodeListi, i);
          const auto& Hi = H(nodeListi, i);
          const auto  mi = m(nodeListi, i);
          const auto  rhoi = rho(nodeListi, i);

          const auto& posj = pos(nodeListj, j);
          const auto& Hj = H(nodeListj, j);
          const auto  mj = m(nodeListj, j);
          const auto  rhoj = rho(nodeListj, j);

          // Compute the node-node weighting
          auto fweightij = 1.0;
          if (nodeListi != nodeListj) {
            if (GeometryRegistrar::coords() == CoordinateType::RZ){
              ri = abs(posi.y());
              rj = abs(posj.y());
              mRZi = mi/(2.0*M_PI*ri);
              mRZj = mj/(2.0*M_PI*rj);
              fweightij = mRZj*rhoi/(mRZi*rhoj);
            } else {
              fweightij = mj*rhoi/(mi*rhoj);
            }
          }
                                        
          xij = posi - posj;
          etai = Hi*xij;
          etaj = Hj*xij;
          thpt = xij.selfdyad()/(xij.magnitude2() + 1.0e-10);

          std::tie(Wi, gWi) = W.kernelAndGradValue(etai.magnitude(), 1.0);
          gradWi = gWi*Hi*etai.unitVector();

          std::tie(Wj, gWj) = W.kernelAndGradValue(etaj.magnitude(), 1.0);
          gradWj = gWj*Hj*etaj.unitVector();

          // Increment the moments
          zerothMoment_thread(nodeListi, i) += fweightij*    std::abs(gWi);
          zerothMoment_thread(nodeListj, j) += 1.0/fweightij*std::abs(gWj);
          secondMoment_thread(nodeListi, i) += fweightij*    gradWi.magnitude2()*thpt;
          secondMoment_thread(nodeListj, j) += 1.0/fweightij*gradWj.magnitude2()*thpt;
        }
      }

      // Do the thread reduction for zeroth and second moments.
      threadReduceFieldLists<Dimension>(threadStack);

    }  // OMP parallel

    // Finish the moments and measure the new H.
    for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
      const auto nodeListPtr = *(dataBase.fluidNodeListBegin() + nodeListi);
      const auto ni = nodeListPtr->numInternalNodes();
      const auto hmin = nodeListPtr->hmin();
      const auto hmax = nodeListPtr->hmax();
      const auto hminratio = nodeListPtr->hminratio();
      const auto nPerh = nodeListPtr->nodesPerSmoothingScale();

#pragma omp parallel for
      for (auto i = 0u; i < ni; ++i) {
        if (flagNodeDone(nodeListi, i) == 0) {
          zerothMoment(nodeListi, i) = Dimension::rootnu(zerothMoment(nodeListi, i));
          H1(nodeListi, i) = smoothingScaleMethod.newSmoothingScale(H(nodeListi, i),
                                                                    pos(nodeListi, i),
                                                                    zerothMoment(nodeListi, i),
                                                                    secondMoment(nodeListi, i),
                                                                    W,
                                                                    hmin,
                                                                    hmax,
                                                                    hminratio,
                                                                    nPerh,
                                                                    connectivityMap,
                                                                    nodeListi,
                                                                    i);

          // If we are preserving the determinant, do it.
          if (fixDeterminant) {
            H1(nodeListi, i) *= Dimension::rootnu(Hdet1(nodeListi, i)/H1(nodeListi, i).Determinant());
            CHECK(fuzzyEqual(H1(nodeListi, i).Determinant(), Hdet1(nodeListi, i)));
          }

          // Check how much this H has changed.
          const auto H1sqrt = H1(nodeListi, i).sqrt();
          const auto phi = (H1sqrt*H(nodeListi, i).Inverse()*H1sqrt).Symmetric().eigenValues();
          const auto phimin = phi.minElement();
          const auto phimax = phi.maxElement();
          const auto deltaHi = max(abs(phimin - 1.0), abs(phimax - 1.0));
          if (deltaHi <= tolerance) flagNodeDone(nodeListi, i) = 1;
          maxDeltaH = max(maxDeltaH, deltaHi);
        }
      }
    }

    // Assign the new H's.
    H.assignFields(H1);

    // Globally reduce the max H change.
    maxDeltaH = allReduce(maxDeltaH, MPI_MAX, Communicator::communicator());

    // Output the statitics.
    if (Process::getRank() == 0)
      cerr << "iterateIdealH: (iteration, deltaH) = ("
           << itr << ", "
           << maxDeltaH << ")"
           << endl;

  }

  // If we have rescaled the nodes per h, now we have to iterate the H determinant
  // to convergence with the proper nperh.
  if (distinctlyGreaterThan(nPerhForIteration, 0.0)) {

    // Reset the nperh.
    size_t k = 0;
    for (auto nodeListItr = dataBase.fluidNodeListBegin();
         nodeListItr != dataBase.fluidNodeListEnd(); 
         ++nodeListItr, ++k) {
      CHECK(k < nperh0.size());
      //const double nperh = nperh0[k];
//       Field<Dimension, SymTensor>& Hfield = **(H.fieldForNodeList(**nodeListItr));
//       Hfield *= Dimension::rootnu(nPerhForIteration/nperh);
      (*nodeListItr)->nodesPerSmoothingScale(nperh0[k]);
    }
    CHECK(k == nperh0.size());

  }

  // If we're fixing the determinant, restore them.
  if (fixDeterminant) {
    for (auto itr = dataBase.internalNodeBegin();
         itr != dataBase.internalNodeEnd();
         ++itr) {
      H(itr) *= Dimension::rootnu(Hdet0(itr)/H(itr).Determinant());
      ENSURE(fuzzyEqual(H(itr).Determinant(), Hdet0(itr), 1.0e-10));
    }
  }

  // Leave the boundary conditions properly enforced.
  for (auto nodeListItr = dataBase.fluidNodeListBegin();
       nodeListItr != dataBase.fluidNodeListEnd(); 
       ++nodeListItr) {
    (*nodeListItr)->numGhostNodes(0);
    (*nodeListItr)->neighbor().updateNodes();
  }
  for (auto boundaryItr = boundaries.begin(); 
       boundaryItr != boundaries.end();
       ++boundaryItr) {
    (*boundaryItr)->setAllGhostNodes(dataBase);
    (*boundaryItr)->finalizeGhostBoundary();
    for (typename DataBase<Dimension>::FluidNodeListIterator nodeListItr = dataBase.fluidNodeListBegin();
         nodeListItr != dataBase.fluidNodeListEnd(); 
         ++nodeListItr) {
      (*nodeListItr)->neighbor().updateNodes();
    }
  }

  for (auto boundaryItr = boundaries.begin(); 
       boundaryItr != boundaries.end();
       ++boundaryItr) {
    (*boundaryItr)->applyFieldListGhostBoundary(m);
  }
  for (auto boundaryItr = boundaries.begin(); 
       boundaryItr != boundaries.end();
       ++boundaryItr) {
    (*boundaryItr)->finalizeGhostBoundary();
  }

  // Report the final timing.
  const auto t1 = clock();
  if (Process::getRank() == 0)
    cerr << "iterateIdealH: required a total of "
         << (t1 - t0)/CLOCKS_PER_SEC
         << " seconds."
         << endl;
}

}

