//---------------------------------Spheral++----------------------------------//
// TensorDamageModel -- Base class for the tensor damage physics models.
// This class does not know how to seed the flaw distribution -- that is 
// required of descendant classes.
//
// References:
//   Benz, W. & Asphaug, E., 1995 "Computer Physics Comm.", 87, 253-265.
//   Benz, W. & Asphaug, E., 1994 "Icarus", 107, 98-116.
//   Randles, P.W. & Libersky, L.D., 1996, "Comput. Methods Appl. Engrg, 
//     139, 375-408
//
// Created by JMO, Thu Sep 29 15:42:05 PDT 2005
//----------------------------------------------------------------------------//
#include "FileIO/FileIO.hh"
#include "TensorDamageModel.hh"
#include "TensorStrainPolicy.hh"
#include "TensorDamagePolicy.hh"
#include "DamageGradientPolicy.hh"
#include "Strength/SolidFieldNames.hh"
#include "NodeList/SolidNodeList.hh"
#include "DataBase/DataBase.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/ReplaceState.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Field/FieldList.hh"
#include "Boundary/Boundary.hh"
#include "Neighbor/Neighbor.hh"

#include <string>
#include <vector>
#include <algorithm>
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

//------------------------------------------------------------------------------
// Constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
TensorDamageModel<Dimension>::
TensorDamageModel(SolidNodeList<Dimension>& nodeList,
                  const TensorStrainAlgorithm strainAlgorithm,
                  const DamageCouplingAlgorithm damageCouplingAlgorithm,
                  const TableKernel<Dimension>& W,
                  const double crackGrowthMultiplier,
                  const double criticalDamageThreshold,
                  const bool damageInCompression,
                  const FlawStorageType& flaws):
  DamageModel<Dimension>(nodeList, W, crackGrowthMultiplier, damageCouplingAlgorithm, flaws),
  mStrain(SolidFieldNames::strainTensor, nodeList),
  mEffectiveStrain(SolidFieldNames::effectiveStrainTensor, nodeList),
  mDdamageDt(TensorDamagePolicy<Dimension>::prefix() + SolidFieldNames::scalarDamage, nodeList),
  mStrainAlgorithm(strainAlgorithm),
  mCriticalDamageThreshold(criticalDamageThreshold),
  mDamageInCompression(damageInCompression) {
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
TensorDamageModel<Dimension>::
~TensorDamageModel() {
}

//------------------------------------------------------------------------------
// Evaluate derivatives.
// For this model we evaluate the derivative of the tensor damage field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
TensorDamageModel<Dimension>::
evaluateDerivatives(const Scalar time,
                    const Scalar dt,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivs) const {

  // Set the scalar magnitude of the damage evolution.
  const auto* nodeListPtr = &(this->nodeList());
  auto&       DDDt = derivs.field(state.buildFieldKey(TensorDamagePolicy<Dimension>::prefix() + SolidFieldNames::scalarDamage, nodeListPtr->name()), 0.0);
  this->computeScalarDDDt(dataBase,
                          state,
                          time,
                          dt,
                          DDDt);
}

//------------------------------------------------------------------------------
// Vote on a time step.
//------------------------------------------------------------------------------
template<typename Dimension>
typename TensorDamageModel<Dimension>::TimeStepType
TensorDamageModel<Dimension>::
dt(const DataBase<Dimension>& /*dataBase*/, 
   const State<Dimension>& /*state*/,
   const StateDerivatives<Dimension>& /*derivs*/,
   const Scalar /*currentTime*/) const {

  // // Look at how quickly we're trying to change the damage.
  // double dt = DBL_MAX;
  // const Field<Dimension, SymTensor>& damage = this->nodeList().damage();
  // const ConnectivityMap<Dimension>& connectivityMap = dataBase.connectivityMap();
  // const vector<const NodeList<Dimension>*>& nodeLists = connectivityMap.nodeLists();
  // const size_t nodeListi = distance(nodeLists.begin(), find(nodeLists.begin(), nodeLists.end(), &(this->nodeList())));
  // for (typename ConnectivityMap<Dimension>::const_iterator iItr = connectivityMap.begin(nodeListi);
  //      iItr != connectivityMap.end(nodeListi);
  //      ++iItr) {
  //   const int i = *iItr;
  //   const double D0 = damage(i).Trace() / Dimension::nDim;
  //   dt = min(dt, 0.8*max(D0, 1.0 - D0)/
  //            std::sqrt(mDdamageDt(i)*mDdamageDt(i) + 1.0e-20));
  // }
  // return TimeStepType(dt, "Rate of damage change");

  return TimeStepType(1.0e100, "Rate of damage change -- NO VOTE.");
}

//------------------------------------------------------------------------------
// Register our state.
//------------------------------------------------------------------------------
template<typename Dimension>
void
TensorDamageModel<Dimension>::
registerState(DataBase<Dimension>& dataBase,
              State<Dimension>& state) {

  typedef typename State<Dimension>::KeyType Key;
  typedef typename State<Dimension>::PolicyPointer PolicyPointer;

  // Register the strain and effective strain.
  PolicyPointer effectiveStrainPolicy(new TensorStrainPolicy<Dimension>(mStrainAlgorithm));
  state.enroll(mStrain);
  state.enroll(mEffectiveStrain, effectiveStrainPolicy);

  // Register the damage.
  // Note we are overriding the default no-op policy for the damage
  // as originally registered by the SolidSPHHydroBase class.
  PolicyPointer damagePolicy(new TensorDamagePolicy<Dimension>(*this));
  state.enroll(this->nodeList().damage(), damagePolicy);

  // Mask out nodes beyond the critical damage threshold from setting the timestep.
  Key maskKey = state.buildFieldKey(HydroFieldNames::timeStepMask, this->nodeList().name());
  Field<Dimension, int>& mask = state.field(maskKey, 0);
  const Field<Dimension, SymTensor>& damage = this->nodeList().damage();
  for (auto i = 0u; i < this->nodeList().numInternalNodes(); ++i) {
    if (damage(i).Trace() > mCriticalDamageThreshold) mask(i) = 0;
  }

  // Register the base classes stuff.
  DamageModel<Dimension>::registerState(dataBase, state);
}

//------------------------------------------------------------------------------
// Register the derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
TensorDamageModel<Dimension>::
registerDerivatives(DataBase<Dimension>& /*dataBase*/,
                    StateDerivatives<Dimension>& derivs) {
  derivs.enroll(mDdamageDt);
}

//------------------------------------------------------------------------------
// Apply the boundary conditions to the ghost nodes.
//------------------------------------------------------------------------------
template<typename Dimension>
void
TensorDamageModel<Dimension>::
applyGhostBoundaries(State<Dimension>& state,
                     StateDerivatives<Dimension>& /*derivs*/) {

  // Grab this models damage field from the state.
  typedef typename State<Dimension>::KeyType Key;
  const Key nodeListName = this->nodeList().name();
  const Key DKey = state.buildFieldKey(SolidFieldNames::tensorDamage, nodeListName);
  CHECK(state.registered(DKey));
  Field<Dimension, SymTensor>& D = state.field(DKey, SymTensor::zero);

  // Apply ghost boundaries to the damage.
  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin();
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
    (*boundaryItr)->applyGhostBoundary(D);
  }
}

//------------------------------------------------------------------------------
// Enforce boundary conditions for the physics specific fields.
//------------------------------------------------------------------------------
template<typename Dimension>
void
TensorDamageModel<Dimension>::
enforceBoundaries(State<Dimension>& state,
                  StateDerivatives<Dimension>& /*derivs*/) {

  // Grab this models damage field from the state.
  typedef typename State<Dimension>::KeyType Key;
  const Key nodeListName = this->nodeList().name();
  const Key DKey = state.buildFieldKey(SolidFieldNames::tensorDamage, nodeListName);
  CHECK(state.registered(DKey));
  Field<Dimension, SymTensor>& D = state.field(DKey, SymTensor::zero);

  // Enforce!
  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
       boundaryItr != this->boundaryEnd();
       ++boundaryItr) {
    (*boundaryItr)->enforceBoundary(D);
  }
}


// //------------------------------------------------------------------------------
// // Initialize at beginning of a step.
// //------------------------------------------------------------------------------
// template<typename Dimension>
// void
// TensorDamageModel<Dimension>::
// initialize(const typename Dimension::Scalar& time,
//            const typename Dimension::Scalar& dt,
//            const DataBase<Dimension>& dataBase,
//            State<Dimension>& state,
//            StateDerivatives<Dimension>& derivs) {

//   // We need to fill in the effective damage and perhaps the damage gradient.
//   // First, grab the pertinent state.
//   typedef typename State<Dimension>::FieldKeyType KeyType;
//   const SolidNodeList<Dimension>& nodeList = this->nodeList();
//   const Field<Dimension, Scalar>& mass = state.scalarField(KeyType(nodeListPtr, HydroFieldNames::mass));
//   const Field<Dimension, Vector>& position = state.vectorField(KeyType(nodeListPtr, HydroFieldNames::position));
//   const Field<Dimension, Scalar>& rho = state.scalarField(KeyType(nodeListPtr, HydroFieldNames::massDensity));
//   const Field<Dimension, Scalar>& weight = state.scalarField(KeyType(nodeListPtr, HydroFieldNames::weight));
//   const Field<Dimension, SymTensor>& H = state.symTensorField(KeyType(nodeListPtr, HydroFieldNames::H));
//   const Field<Dimension, Scalar>& omega = state.scalarField(KeyType(nodeListPtr, HydroFieldNames::omegaGradh));
//   const Field<Dimension, SymTensor>& D = state.symTensorField(KeyType(nodeListPtr, SolidFieldNames::tensorDamage));

//   Field<Dimension, SymTensor>& Deff = state.symTensorField(KeyType(nodeListPtr, SolidFieldNames::effectiveTensorDamage));
//   Field<Dimension, Vector>& gradD = derivs.vectorField(KeyType(&nodeList, SolidFieldNames::gradDamage));

//   // Get the kernel.
//   const TableKernel<Dimension>& W = nodeListPtr->kernel();
//   const double etaMax2 = W.kernelExtent() * W.kernelExtent();

//   // The neighbor connectivity.
//   const ConnectivityMap<Dimension>& connectivityMap = dataBase.connectivityMap();
//   const vector<const NodeList<Dimension>*>& nodeLists = connectivityMap.nodeLists();
//   const size_t nodeListi = distance(nodeLists.begin(), find(nodeLists.begin(), nodeLists.end(), nodeListPtr));
//   CHECK(nodeListi < nodeLists.size());

//   const size_t firstGhostNode = nodeListPtr->firstGhostNode();

//   // Iterate over the internal nodes in this NodeList.
//   Field<Dimension, Scalar> normaliation("normalization", *nodeListPtr, 0.0);
//   Deff = D;
//   for (size_t i = 0; i != nodeListPtr->numInternalNodes(); ++i) {

//     // State for node i.
//     const Scalar& mi = mass(i);
//     const Vector& ri = position(i);
//     const Scalar& rhoi = rho(i);
//     const Scalar& weighti = weight(i);
//     const SymTensor& Hi = H(i);
//     const Scalar Hdeti = Hi.Determinant();
//     const Scalar safeOmegai = omega(i)/(omega(i)*omega(i) + 1.0e-20);
//     const SymTensor& Di = D(i);
//     const Scalar Dmagi = Di.Trace();
//     CHECK(mi > 0.0);
//     CHECK(rhoi > 0.0);
//     CHECK(weighti > 0.0);
//     CHECK(safeOmegai > 0.0);
//     CHECK(Hdeti > 0.0);
//     CHECK(Dmagi >= 0.0 and Dmagi <= Dimension::nDim);

//     SymTensor& Deffi = Deff(i);
//     Vector& gradDi = gradD(i);

//     // Get the connectivity info for this node.  We only need to proceed if
//     // there are some nodes in this list.
//     const vector<int>& connectivity = connectivityMap.connectivityForNode(nodeListPtr, i)[nodeListi];
//     if (connectivity.size() > 0) {

//       // Iterate over the neighbors.
//       for (vector<int>::const_iterator jItr = connectivity.begin();
//            jItr != connectivity.end();
//            ++jItr) {
//         const int j = *jItr;

//         // Only proceed if this node pair has not been calculated yet.
//         if (j > i) {

//           // Get the state for node j
//           const Scalar& mj = mass(j);
//           const Vector& rj = position(j);
//           const Scalar& rhoj = rho(j);
//           const Scalar& weightj = weight(j);
//           const SymTensor& Hj = H(j);
//           const Scalar Hdetj = Hj.Determinant();
//           const Scalar safeOmegaj = omega(j)/(omega(j)*omega(j) + 1.0e-20);
//           const SymTensor& Dj = D(j);
//           const Scalar& Dmagj = Dj.Trace();
//           CHECK(mj > 0.0);
//           CHECK(rhoj > 0.0);
//           CHECK(weightj > 0.0);
//           CHECK(safeOmegaj > 0.0);
//           CHECK(Hdetj > 0.0);
//           CHECK(Dmagj >= 0.0 and Dmagj <= Dimension::nDim);

//           SymTensor& Deffj = Deff(j);
//           Vector& gradDnewj = gradD(j);

//           // Node displacement and weighting.
//           const Vector rij = ri - rj;

//           const Vector etai = Hi*rij;
//           const Scalar etaMagi = etai.magnitude();
//           const std::pair<double, double> WWi = W.kernelAndGradValue(etaMagi, Hdeti);
//           const Scalar Wi = WWi.first;
//           const Vector gradWi = WWi.second * (Hi*etai.unitVector());

//           const Vector etaj = Hj*rij;
//           const Scalar etaMagj = etaj.magnitude();
//           const std::pair<double, double> WWj = W.kernelAndGradValue(etaMagj, Hdetj);
//           const Scalar Wj = WWj.first;
//           const Vector gradWj = WWj.second * (Hj*etaj.unitVector());


//           const Vector gradWj = W.grad(etaMagj, Hdetj) * (Hj*etaj.unitVector());

//           // Increment the effective damage.
//           switch(mEffDamageAlgorithm) {
//           case Max:
//             if (Dmagj > Dmaji) Di = Dj;
//             if (Dmagi > Dmajj) Dj = Di;
//             break;

//           case Sampled:
//             normalizationi += Dmagj*Wi;
//             normalizationj += Dmagi*Wj;
//             Di += Dmagj*Wi * Dj;
//             Dj += Dmagi*Wj * Di;
//             break;
//           }

//           // Increment the gradient.
//           gradDi += mj*(Dj - Di)*gradWi;
//           gradDj += mi*(Dj - Di)*gradWj;

//         }
//       }
//     }

//     // Finish the effective damage.
//     if (mEffDamageAlgorithm == Sampled) {
//       CHECK(normalizationi >= 0.0);
//       Deffi /= normalizationi + tiny;
//     }

//     // Finish the gradient.
//     if (mUseDamageGradient) {
//       CHECK(rhoi > 0.0);
//       gradDi /= rhoi;
//     } else {
//       gradDi.Zero();
//     }

//   }










//   // If needed, compute the gradient of the damage.
//   if (mUseDamageGradient) {

//     // State fields.
//     typedef typename State<Dimension>::FieldKeyType KeyType;
//     const SolidNodeList<Dimension>& nodeList = this->nodeList();
//     const Field<Dimension, Scalar>& mass = state.scalarField(KeyType(&nodeList, HydroFieldNames::mass));
//     const Field<Dimension, Vector>& position = state.vectorField(KeyType(&nodeList, HydroFieldNames::position));
//     const Field<Dimension, Scalar>& rho = state.scalarField(KeyType(&nodeList, HydroFieldNames::massDensity));
//     const Field<Dimension, SymTensor>& H = state.symTensorField(KeyType(&nodeList, HydroFieldNames::H));
//     const Field<Dimension, Scalar>& omega = state.scalarField(KeyType(nodeListPtr, HydroFieldNames::omegaGradh));
//     const Field<Dimension, Scalar>& D = state.scalarField(KeyType(&nodeList, SolidFieldNames::tensorDamage));
//     Field<Dimension, Vector>& gradD = derivs.vectorField(KeyType(&nodeList, SolidFieldNames::gradDamage));

//     // Get the kernel.
//     const TableKernel<Dimension>& W = nodeList.kernel();
//     const double etaMax2 = W.kernelExtent() * W.kernelExtent();

//     // The neighbor connectivity.
//     const ConnectivityMap<Dimension>& connectivityMap = dataBase.connectivityMap();
//     const vector<const NodeList<Dimension>*>& nodeLists = connectivityMap.nodeLists();
//     const int nodeListi = distance(nodeLists.begin(), find(nodeLists.begin(), nodeLists.end(), &nodeList));
//     CHECK(nodeListi < nodeLists.size());

//     // Initialize the grad field.
//     gradD.Zero();

//     // Iterate over the nodes in this NodeList.
//     for (size_t i = 0; i != nodeList.numInternalNodes(); ++i) {

//       // State for node i.
//       const Vector& ri = position(i);
//       const Scalar& rhoi = rho(i);
//       const SymTensor& Hi = H(i);
//       const Scalar& Di = D(i);

//       // Get the connectivity for this node.
//       const vector< vector<int> >& fullConnectivity = connectivityMap.connectivityForNode(&nodeList, i);
//       CHECK(fullConnectivity.size() == nodeLists.size());
//       const vector<int>& connectivity = fullConnectivity[nodeListi];

//       // Iterate over the neighbor nodes in this NodeList.
//       for (typename vector<int>::const_iterator jItr = connectivity.begin();
//            jItr != connectivity.end();
//            ++jItr) {
//         const int j = *jItr;
     
//         // State for node j.
//         const Scalar& mj = mass(j);
//         const Vector& rj = position(j);
//         const Scalar& Dj = D(j);

//         // Node displacement and weighting.
//         const Vector rij = ri - rj;
//         const Vector etai = Hi*rij;
//         const Vector gradWi = Hi*W.grad(etai.magnitude(), Hi.Determinant()) * etai.unitVector();

//         // Increment the gradient.
//         gradD(i) += mj*(Dj - Di)*gradWi;
//       }

//       // Finalize the gradient.
//       CHECK(rhoi > 0.0);
//       gradD(i) /= rhoi;

//     }

//     // Apply boundary conditions to the gradient.
//     for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
//          boundaryItr != this->boundaryEnd();
//          ++boundaryItr) {
//       (*boundaryItr)->applyGhostBoundary(gradD);
//     }
//     for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
//          boundaryItr != this->boundaryEnd();
//          ++boundaryItr) {
//       (*boundaryItr)->finalizeGhostBoundary();
//     }

//   }

// }

//------------------------------------------------------------------------------
// Dump the current state to the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
TensorDamageModel<Dimension>::
dumpState(FileIO& file, const string& pathName) const {
  DamageModel<Dimension>::dumpState(file, pathName);
  file.write(mStrain, pathName + "/strain");
  file.write(mEffectiveStrain, pathName + "/effectiveStrain");
  file.write(mDdamageDt, pathName + "/DdamageDt");
}

//------------------------------------------------------------------------------
// Restore the state from the given file.
//------------------------------------------------------------------------------
template<typename Dimension>
void
TensorDamageModel<Dimension>::
restoreState(const FileIO& file, const string& pathName) {
  DamageModel<Dimension>::restoreState(file, pathName);
  file.read(mStrain, pathName + "/strain");
  file.read(mEffectiveStrain, pathName + "/effectiveStrain");
  file.read(mDdamageDt, pathName + "/DdamageDt");
}

}

