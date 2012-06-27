//------------------------------------------------------------------------------
// Compute the CSPH grad h correction due to Springel et al.
//------------------------------------------------------------------------------
#include "computeCSPHCorrections.hh"
#include "Field/Field.hh"
#include "Field/FieldList.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Kernel/TableKernel.hh"
#include "NodeList/NodeList.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Geometry/outerProduct.hh"
#include "Geometry/innerProduct.hh"

namespace Spheral {
namespace CSPHSpace {

using namespace std;
using std::min;
using std::max;
using std::abs;

using FieldSpace::Field;
using FieldSpace::FieldList;
using NeighborSpace::ConnectivityMap;
using KernelSpace::TableKernel;
using NodeSpace::NodeList;
using Geometry::outerProduct;
using Geometry::innerProduct;

template<typename Dimension>
void
computeCSPHCorrections(const ConnectivityMap<Dimension>& connectivityMap,
                       const TableKernel<Dimension>& W,
                       const FieldList<Dimension, typename Dimension::Scalar>& weight,
                       const FieldList<Dimension, typename Dimension::Vector>& position,
                       const FieldList<Dimension, typename Dimension::SymTensor>& H,
                       FieldList<Dimension, typename Dimension::Scalar>& A,
                       FieldList<Dimension, typename Dimension::Vector>& B,
                       FieldList<Dimension, typename Dimension::Vector>& C,
                       FieldList<Dimension, typename Dimension::Tensor>& D,
                       FieldList<Dimension, typename Dimension::Vector>& gradA,
                       FieldList<Dimension, typename Dimension::Tensor>& gradB) {

  // TAU timers.
  TAU_PROFILE("computeCSPHCorrections", "", TAU_USER);

  // Pre-conditions.
  const size_t numNodeLists = A.size();
  REQUIRE(weight.size() == numNodeLists);
  REQUIRE(position.size() == numNodeLists);
  REQUIRE(H.size() == numNodeLists);
  REQUIRE(B.size() == numNodeLists);
  REQUIRE(C.size() == numNodeLists);
  REQUIRE(D.size() == numNodeLists);
  REQUIRE(gradA.size() == numNodeLists);
  REQUIRE(gradB.size() == numNodeLists);

  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;
  typedef typename Dimension::ThirdRankTensor ThirdRankTensor;

  // Zero out the result.
  A = 0.0;
  B = Vector::zero;
  C = Vector::zero;
  D = Tensor::zero;
  gradA = Vector::zero;
  gradB = Tensor::zero;

  // Walk the FluidNodeLists.
  for (size_t nodeListi = 0; nodeListi != numNodeLists; ++nodeListi) {
    const NodeList<Dimension>& nodeList = A[nodeListi]->nodeList();
    const int firstGhostNodei = nodeList.firstGhostNode();

    // We can derive everything in terms of the zeroth, first, and second moments 
    // of the local positions.
    Field<Dimension, Scalar> m0("zeroth moment", nodeList);
    Field<Dimension, Vector> m1("first moment", nodeList);
    Field<Dimension, SymTensor> m2("second moment", nodeList);
    Field<Dimension, Vector> gradm0("gradient of the zeroth moment", nodeList);
    Field<Dimension, Tensor> gradm1("gradient of the first moment", nodeList);
    Field<Dimension, ThirdRankTensor> gradm2("gradient of the second moment", nodeList);

    // Iterate over the nodes in this node list.
    for (typename ConnectivityMap<Dimension>::const_iterator iItr = connectivityMap.begin(nodeListi);
         iItr != connectivityMap.end(nodeListi);
         ++iItr) {
      const int i = *iItr;

      // Get the state for node i.
      const Scalar wi = weight(nodeListi, i);
      const Vector& ri = position(nodeListi, i);
      const SymTensor& Hi = H(nodeListi, i);
      const Scalar Hdeti = Hi.Determinant();

      // Self contribution.
      const Scalar wwi = wi*W(0.0, Hdeti);
      m0(i) += wwi;
      gradm1(i) += wwi;

      // Neighbors!
      const vector<vector<int> >& fullConnectivity = connectivityMap.connectivityForNode(nodeListi, i);
      CHECK(fullConnectivity.size() == numNodeLists);
      const vector<int>& connectivity = fullConnectivity[nodeListi];

      // Iterate over the neighbors for in this NodeList.
      for (vector<int>::const_iterator jItr = connectivity.begin();
           jItr != connectivity.end();
           ++jItr) {
        const int j = *jItr;

        // Check if this node pair has already been calculated.
        if (connectivityMap.calculatePairInteraction(nodeListi, i, 
                                                     nodeListi, j,
                                                     firstGhostNodei)) {

          // State of node j.
          const Scalar wj = weight(nodeListi, j);
          const Vector& rj = position(nodeListi, j);
          const SymTensor& Hj = H(nodeListi, j);
          const Scalar Hdetj = Hj.Determinant();

          // Kernel weighting and gradient.
          const Vector rij = ri - rj;
          const Vector etai = Hi*rij;
          const Vector etaj = Hj*rij;
          const std::pair<double, double> WWi = W.kernelAndGradValue(etai.magnitude(), Hdeti);
          const Scalar& Wi = WWi.first;
          const Vector gradWi = -(Hi*etai.unitVector())*WWi.second;
          const std::pair<double, double> WWj = W.kernelAndGradValue(etaj.magnitude(), Hdetj);
          const Scalar& Wj = WWj.first;
          const Vector gradWj = (Hj*etaj.unitVector())*WWj.second;

          // Zeroth moment. 
          const Scalar wwi = wi*Wi;
          const Scalar wwj = wj*Wj;
          m0(i) += wwj;
          m0(j) += wwi;
          gradm0(i) += wj*gradWj;
          gradm0(j) += wi*gradWi;

          // First moment. 
          m1(i) += wwj * rij;
          m1(j) -= wwi * rij;
          gradm1(i) += wj*( rij*gradWj + Tensor::one*Wj);
          gradm1(j) += wi*(-rij*gradWi + Tensor::one*Wi);

          // Second moment.
          const SymTensor thpt = rij.selfdyad();
          m2(i) += wwj*thpt;
          m2(j) += wwi*thpt;
          gradm2(i) += wj*outerProduct<Dimension>(thpt, gradWj);
          gradm2(j) += wi*outerProduct<Dimension>(thpt, gradWi);
          for (size_t ii = 0; ii != Dimension::nDim; ++ii) {
            for (size_t jj = 0; jj != Dimension::nDim; ++jj) {
              gradm2(i)(ii, jj, jj) += wwj*rij(ii);
              gradm2(j)(ii, jj, jj) -= wwi*rij(ii);

              gradm2(i)(ii, jj, ii) += wwj*rij(jj);
              gradm2(j)(ii, jj, ii) -= wwi*rij(jj);
            }
          }
        }
      }

      // Based on the moments we can calculate the CSPH corrections terms and their gradients.
      if (i < firstGhostNodei) {
        CHECK2(abs(m2(i).Determinant()) > 1.0e-30, i << " " << m0(i) << " " << m2(i).Determinant());
        const SymTensor m2inv = m2(i).Inverse();
        const Vector m2invm1 = m2inv*m1(i);
        const Scalar Ainv = m0(i) - m2invm1.dot(m1(i));
        CHECK(Ainv != 0.0);
        A(nodeListi, i) = 1.0/Ainv;
        B(nodeListi, i) = -m2invm1;
        gradA(nodeListi, i) = -A(nodeListi, i)*A(nodeListi, i)*
          (gradm0(i) + innerProduct<Dimension>(innerProduct<Dimension>(innerProduct<Dimension>(m2inv, gradm2(i)), m2inv), m1(i)).dot(m1(i)) -
           innerProduct<Dimension>(m1(i), m2inv*gradm1(i)) - innerProduct<Dimension>(m2inv*m1(i), gradm1(i)));
        gradB(nodeListi, i) = m2inv*(innerProduct<Dimension>(innerProduct<Dimension>(gradm2(i), m2inv), m1(i)) - gradm1(i));
      }
    }
  }

  // Note -- we are suspending the next order corrections for now!

  // Now we have the information to compute the "integration correction" (a.k.a C).
  // This is actually an iterative process (ewwww!)
//   {
//     const int maxIterations = 10;
//     const double tol = 1.0e-10;
//     int iter = 0;
//     double variation = 2.0*tol;
//     while (iter < maxIterations and variation > tol) {
//       ++iter;
//       variation = 0.0;

//       Field<Dimension, Vector> Cnew("new value for C", *nodeListPtr);
//       for (int i = 0; i != nodeListPtr->numInternalNodes(); ++i) {
//         const Scalar& wi = weight(i);
//         const Vector& ri = position(i);
//         const SymTensor& Hi = H(i);
//         const Scalar Hdeti = Hi.Determinant();
//         const Scalar& Ai = A(i);
//         const Vector& Bi = B(i);
//         const Vector& Ci = C(i);
//         const Vector& gradAi = gradA(i);
//         const Tensor& gradBi = gradB(i);

//         // Self contribution.
//         Scalar Wj, gWj;
//         Vector gradWj;
//         CSPHKernelAndGradient(W, Vector(), Vector(), Hi, Hdeti, Ai, Bi, gradAi, gradBi, Wj, gWj, gradWj);
//         Cnew(i) += wi*(Ci*Wj - gradWj);

//         // Get the neighbors for this node (in this NodeList).
//         const vector<int>& connectivity = connectivityMap.connectivityForNode(nodeListPtr, i)[nodeListi];
//         for (vector<int>::const_iterator jItr = connectivity.begin();
//              jItr != connectivity.end();
//              ++jItr) {
//           const int j = *jItr;

//           // Check if this node pair has already been calculated.
// //           if (j > i) {
//             const Scalar& wj = weight(j);
//             const Vector& rj = position(j);
//             const SymTensor& Hj = H(j);
//             const Scalar Hdetj = Hj.Determinant();
//             const Scalar& Aj = A(j);
//             const Vector& Bj = B(j);
//             const Vector& Cj = C(j);
//             const Vector& gradAj = gradA(j);
//             const Tensor& gradBj = gradB(j);

//             // Kernel weighting and gradient.
//             const Vector rji = rj - ri;
//             const Vector etai = Hi*rji;
//             const Vector etaj = Hj*rji;
//             Scalar Wi, gWi, Wj, gWj;
//             Vector gradWi, gradWj;
//             CSPHKernelAndGradient(W, rji, etaj, Hj, Hdetj, Ai, Bi, gradAi, gradBi, Wj, gWj, gradWj);
//             CSPHKernelAndGradient(W, rji, etai, Hi, Hdeti, Aj, Bj, gradAj, gradBj, Wi, gWi, gradWi);
//             Cnew(i) += wj*(Cj*Wj - gradWj);
// //             Cnew(j) += wi*(Ci*Wi + gradWi);
// //           }

// //             // HACK!  Boundary contribution.
// //             if (j == 0 or j == 99) {
// //               const double AAj = (j == 0) ? -1.0 : 1.0;
// //               Cnew(i) += AAj*Wj;
// //             }
// //             // HACK!  Boundary contribution.

//         }

//         // Update the maximum variation.
//         variation = max(variation, (Cnew(i) - C(i)).magnitude()/max(1.0, sqrt(abs(C(i).dot(Cnew(i))))));
//       }

//       for (int i = 0; i != 10; ++i) cerr << Cnew(i) << " ";
//       cerr << endl;

//       // Set the C field to the new value.
//       C = Cnew;

//       // Get the global max variation.
//       variation = safeAllReduceMax(variation);
//       cerr << "Iteration " << iter << " : variation " << variation << endl;
//     }

//     // Check if we converged or now.
//     cerr << "Required " << iter << " iterations to converge to " << variation << endl;
//     if (variation > tol) cerr << "CSPHFluidDerivatives::updateCorrections failed to converge for C" << endl;

//   }

}

}
}

//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "Geometry/Dimension.hh"
namespace Spheral {
  namespace CSPHSpace {
    template void computeCSPHCorrections(const ConnectivityMap<Dim<1> >&, 
                                         const TableKernel<Dim<1> >&, 
                                         const FieldList<Dim<1>, Dim<1>::Scalar>&,
                                         const FieldList<Dim<1>, Dim<1>::Vector>&,
                                         const FieldList<Dim<1>, Dim<1>::SymTensor>&,
                                         FieldList<Dim<1>, Dim<1>::Scalar>&,
                                         FieldList<Dim<1>, Dim<1>::Vector>&,
                                         FieldList<Dim<1>, Dim<1>::Vector>&,
                                         FieldList<Dim<1>, Dim<1>::Tensor>&,
                                         FieldList<Dim<1>, Dim<1>::Vector>&,
                                         FieldList<Dim<1>, Dim<1>::Tensor>&);
    template void computeCSPHCorrections(const ConnectivityMap<Dim<2> >&, 
                                         const TableKernel<Dim<2> >&, 
                                         const FieldList<Dim<2>, Dim<2>::Scalar>&,
                                         const FieldList<Dim<2>, Dim<2>::Vector>&,
                                         const FieldList<Dim<2>, Dim<2>::SymTensor>&,
                                         FieldList<Dim<2>, Dim<2>::Scalar>&,
                                         FieldList<Dim<2>, Dim<2>::Vector>&,
                                         FieldList<Dim<2>, Dim<2>::Vector>&,
                                         FieldList<Dim<2>, Dim<2>::Tensor>&,
                                         FieldList<Dim<2>, Dim<2>::Vector>&,
                                         FieldList<Dim<2>, Dim<2>::Tensor>&);
    template void computeCSPHCorrections(const ConnectivityMap<Dim<3> >&, 
                                         const TableKernel<Dim<3> >&, 
                                         const FieldList<Dim<3>, Dim<3>::Scalar>&,
                                         const FieldList<Dim<3>, Dim<3>::Vector>&,
                                         const FieldList<Dim<3>, Dim<3>::SymTensor>&,
                                         FieldList<Dim<3>, Dim<3>::Scalar>&,
                                         FieldList<Dim<3>, Dim<3>::Vector>&,
                                         FieldList<Dim<3>, Dim<3>::Vector>&,
                                         FieldList<Dim<3>, Dim<3>::Tensor>&,
                                         FieldList<Dim<3>, Dim<3>::Vector>&,
                                         FieldList<Dim<3>, Dim<3>::Tensor>&);
  }
}

