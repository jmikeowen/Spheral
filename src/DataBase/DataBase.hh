//---------------------------------Spheral++----------------------------------//
// DataBase -- The central point to store NodeLists and FieldLists, to serve the 
//             Spheral++ physics modules.
//
// Created by JMO, Sun Feb  6 13:44:49 PST 2000
//----------------------------------------------------------------------------//
#ifndef DataBase_HH
#define DataBase_HH

#include "NodeList/NodeList.hh"
#include "NodeList/FluidNodeList.hh"
#include "NodeList/SolidNodeList.hh"
#include "Field/NodeIterators.hh"
#include "Neighbor/ConnectivityMap.hh"

#include <vector>

namespace Spheral {
  template<typename Dimension, typename DataType> class Field;
  template<typename Dimension, typename DataType> class FieldList;
  template<typename Dimension> class TableKernel;
}

namespace Spheral {

template<typename Dimension>
class DataBase {
public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;

  typedef typename std::vector<NodeList<Dimension>*>::iterator NodeListIterator;
  typedef typename std::vector<NodeList<Dimension>*>::const_iterator ConstNodeListIterator;

  typedef typename std::vector<FluidNodeList<Dimension>*>::iterator FluidNodeListIterator;
  typedef typename std::vector<FluidNodeList<Dimension>*>::const_iterator ConstFluidNodeListIterator;

  typedef typename std::vector<SolidNodeList<Dimension>*>::iterator SolidNodeListIterator;
  typedef typename std::vector<SolidNodeList<Dimension>*>::const_iterator ConstSolidNodeListIterator;

  typedef ConnectivityMap<Dimension> ConnectivityMapType;
  typedef std::shared_ptr<ConnectivityMapType> ConnectivityMapPtr;
  
  // It is convenient to be able to query the DataBase for the problem
  // dimensionality for Python.
  static int nDim;
  static bool isRZ;

  // Constructors.
  DataBase();

  // Destructor.
  ~DataBase();

  // Assignment.
  DataBase& operator=(const DataBase& rhs);

  // Number of NodeLists we have in the DataBase.
  unsigned int numNodeLists() const;
  unsigned int numFluidNodeLists() const;
  int numSolidNodeLists() const;

  // Numbers of nodes.
  int numInternalNodes() const;
  int numGhostNodes() const;
  int numNodes() const;

  int globalNumInternalNodes() const;
  int globalNumGhostNodes() const;
  int globalNumNodes() const;

  // Provide normal iterator methods over the DataBase NodeLists.
  NodeListIterator nodeListBegin();
  NodeListIterator nodeListEnd();

  ConstNodeListIterator nodeListBegin() const;
  ConstNodeListIterator nodeListEnd() const;

  FluidNodeListIterator fluidNodeListBegin();
  FluidNodeListIterator fluidNodeListEnd();

  ConstFluidNodeListIterator fluidNodeListBegin() const;
  ConstFluidNodeListIterator fluidNodeListEnd() const;

  NodeListIterator fluidNodeListAsNodeListBegin();
  NodeListIterator fluidNodeListAsNodeListEnd();

  ConstNodeListIterator fluidNodeListAsNodeListBegin() const;
  ConstNodeListIterator fluidNodeListAsNodeListEnd() const;

  SolidNodeListIterator solidNodeListBegin();
  SolidNodeListIterator solidNodeListEnd();

  ConstSolidNodeListIterator solidNodeListBegin() const;
  ConstSolidNodeListIterator solidNodeListEnd() const;

  NodeListIterator solidNodeListAsNodeListBegin();
  NodeListIterator solidNodeListAsNodeListEnd();

  ConstNodeListIterator solidNodeListAsNodeListBegin() const;
  ConstNodeListIterator solidNodeListAsNodeListEnd() const;

  // Provide NodeIterators to go over the elements of NodeLists/FieldLists.
  AllNodeIterator<Dimension> nodeBegin() const;
  AllNodeIterator<Dimension> nodeEnd() const;

  InternalNodeIterator<Dimension> internalNodeBegin() const;
  InternalNodeIterator<Dimension> internalNodeEnd() const;
  
  GhostNodeIterator<Dimension> ghostNodeBegin() const;
  GhostNodeIterator<Dimension> ghostNodeEnd() const;
  
  MasterNodeIterator<Dimension> masterNodeBegin(const std::vector<std::vector<int>>& masterLists) const;
  MasterNodeIterator<Dimension> masterNodeEnd() const;
  
  CoarseNodeIterator<Dimension> coarseNodeBegin(const std::vector<std::vector<int>>& coarseNeighbors) const;
  CoarseNodeIterator<Dimension> coarseNodeEnd() const;
  
  RefineNodeIterator<Dimension> refineNodeBegin(const std::vector<std::vector<int>>& refineNeighbors) const;
  RefineNodeIterator<Dimension> refineNodeEnd() const;

  // Same iterator methods, but over FluidNodeLists.
  AllNodeIterator<Dimension> fluidNodeBegin() const;
  AllNodeIterator<Dimension> fluidNodeEnd() const;
  
  InternalNodeIterator<Dimension> fluidInternalNodeBegin() const;
  InternalNodeIterator<Dimension> fluidInternalNodeEnd() const;
  
  GhostNodeIterator<Dimension> fluidGhostNodeBegin() const;
  GhostNodeIterator<Dimension> fluidGhostNodeEnd() const;
  
  MasterNodeIterator<Dimension> fluidMasterNodeBegin(const std::vector<std::vector<int>>& masterLists) const;
  MasterNodeIterator<Dimension> fluidMasterNodeEnd() const;
  
  CoarseNodeIterator<Dimension> fluidCoarseNodeBegin(const std::vector<std::vector<int>>& coarseNeighbors) const;
  CoarseNodeIterator<Dimension> fluidCoarseNodeEnd() const;
  
  RefineNodeIterator<Dimension> fluidRefineNodeBegin(const std::vector<std::vector<int>>& refineNeighbors) const;
  RefineNodeIterator<Dimension> fluidRefineNodeEnd() const;

  // Optimize all Neighbor objects for the current state.
  void reinitializeNeighbors() const;

  // Update the internal connectivity map.
  void updateConnectivityMap(const bool computeGhostConnectivity,
                             const bool computeOverlapConnectivity,
                             const bool computeIntersectionConnectivity) const;
  void patchConnectivityMap(const FieldList<Dimension, int>& flags,
                            const FieldList<Dimension, int>& old2new) const;

  // Get the connectivity map.
  const ConnectivityMapType& connectivityMap() const;
  const ConnectivityMapType& connectivityMap(const bool computeGhostConnectivity,
                                             const bool computeOverlapConnectivity,
                                             const bool computeIntersectionConnectivity) const;
  ConnectivityMapPtr connectivityMapPtr(const bool computeGhostConnectivity,
                                        const bool computeOverlapConnectivity,
                                        const bool computeIntersectionConnectivity) const;

  // Methods to add, remove, and verify NodeLists.
  void appendNodeList(SolidNodeList<Dimension>& nodeList);
  void appendNodeList(FluidNodeList<Dimension>& nodeList);
  void appendNodeList(NodeList<Dimension>& nodeList);

  void deleteNodeList(SolidNodeList<Dimension>& nodeList);
  void deleteNodeList(FluidNodeList<Dimension>& nodeList);
  void deleteNodeList(NodeList<Dimension>& nodeList);

  bool haveNodeList(const NodeList<Dimension>& nodeList) const;

  // Allow const access to the list of NodeList pointers.
  const std::vector<NodeList<Dimension>*>& nodeListPtrs() const;
  const std::vector<FluidNodeList<Dimension>*>& fluidNodeListPtrs() const;
  const std::vector<SolidNodeList<Dimension>*>& solidNodeListPtrs() const;

  // Provide convenience functions for manipulating the neighbor information
  // of the NodeLists.
  void setMasterNodeLists(const Vector& position,
                          const SymTensor& H,
                          std::vector<std::vector<int>>& masterLists,
                          std::vector<std::vector<int>>& coarseNeighbors,
                          const bool computeGhostConnectivity) const;
  void setMasterFluidNodeLists(const Vector& position,
                               const SymTensor& H,
                               std::vector<std::vector<int>>& masterLists,
                               std::vector<std::vector<int>>& coarseNeighbors,
                               const bool computeGhostConnectivity) const;

  void setRefineNodeLists(const Vector& position,
                          const SymTensor& H,
                          const std::vector<std::vector<int>>& coarseNeighbors,
                          std::vector<std::vector<int>>& refineNeighbors) const;
  void setRefineFluidNodeLists(const Vector& position,
                               const SymTensor& H,
                               const std::vector<std::vector<int>>& coarseNeighbors,
                               std::vector<std::vector<int>>& refineNeighbors) const;

  // Query methods which return "global" fields (FieldLists) for quantities
  // defined over all NodeLists.  These methods all build up FieldLists
  // referencing internally cached data in NodeLists, and so are fast.
  FieldList<Dimension, Scalar> globalMass() const;
  FieldList<Dimension, Vector> globalPosition() const;
  FieldList<Dimension, Vector> globalVelocity() const;
  FieldList<Dimension, SymTensor> globalHfield() const;
  FieldList<Dimension, Scalar> globalWork() const;

  FieldList<Dimension, Scalar> fluidMass() const;
  FieldList<Dimension, Vector> fluidPosition() const;
  FieldList<Dimension, Vector> fluidVelocity() const;
  FieldList<Dimension, Scalar> fluidMassDensity() const;
  FieldList<Dimension, Scalar> fluidSpecificThermalEnergy() const;
  FieldList<Dimension, SymTensor> fluidHfield() const;
  FieldList<Dimension, Scalar> fluidWork() const;

  FieldList<Dimension, Scalar> solidMass() const;
  FieldList<Dimension, Vector> solidPosition() const;
  FieldList<Dimension, Vector> solidVelocity() const;
  FieldList<Dimension, Scalar> solidMassDensity() const;
  FieldList<Dimension, Scalar> solidSpecificThermalEnergy() const;
  FieldList<Dimension, SymTensor> solidHfield() const;
  FieldList<Dimension, Scalar> solidWork() const;
  FieldList<Dimension, SymTensor> solidDeviatoricStress() const;
  FieldList<Dimension, Scalar> solidPlasticStrain() const;
  FieldList<Dimension, Scalar> solidPlasticStrainRate() const;
  FieldList<Dimension, SymTensor> solidDamage() const;
  FieldList<Dimension, int> solidFragmentIDs() const;
  FieldList<Dimension, int> solidParticleTypes() const;

  // We can also return the node extent Fields stored in the Neighbor objects.
  FieldList<Dimension, Vector> globalNodeExtent() const;
  FieldList<Dimension, Vector> fluidNodeExtent() const;
  FieldList<Dimension, Vector> solidNodeExtent() const;

  // These functions return FieldLists with Fields that have to be calculated and
  // stored, so they are more expensive.
  // These FieldLists represent state that must be computed on the fly.
  void globalHinverse(FieldList<Dimension, SymTensor>& result) const;
  void fluidHinverse(FieldList<Dimension, SymTensor>& result) const;
  void fluidPressure(FieldList<Dimension, Scalar>& result) const;
  void fluidTemperature(FieldList<Dimension, Scalar>& result) const;
  void fluidSoundSpeed(FieldList<Dimension, Scalar>& result) const;
  void fluidVolume(FieldList<Dimension, Scalar>& result) const;
  void fluidGamma(FieldList<Dimension, Scalar>& result) const;
  void fluidEntropy(FieldList<Dimension, Scalar>& result) const;
  void fluidLinearMomentum(FieldList<Dimension, Vector>& result) const;
  void fluidTotalEnergy(FieldList<Dimension, Scalar>& result) const;
  void fluidSpecificHeat(const FieldList<Dimension, Scalar>& temperature,
                         FieldList<Dimension, Scalar>& result) const;

  // Collect the number of neighbors for each node from the ConnectivityMap.
  FieldList<Dimension, int> numNeighbors() const;

  //............................................................................
  // Create new FieldLists of size the number of NodeLists or FluidNodeLists.
  template<typename DataType>
  FieldList<Dimension, DataType> newGlobalFieldList(const DataType value,
                                                    const typename Field<Dimension, DataType>::FieldName name = "Unnamed Field") const;
  template<typename DataType>
  FieldList<Dimension, DataType> newFluidFieldList(const DataType value,
                                                   const typename Field<Dimension, DataType>::FieldName name = "Unnamed Field") const;
  template<typename DataType>
  FieldList<Dimension, DataType> newSolidFieldList(const DataType value,
                                                   const typename Field<Dimension, DataType>::FieldName name = "Unnamed Field") const;

  // Resize a FieldList to the number of NodeLists or FluidNodeLists.
  // Optionally we can also set all elements in the FieldList to the specified value.
  // Note that if the FieldList is resized it is reconstructed from scratch, so all elements
  // will get the new value regardless of resetValues.
  template<typename DataType>
  void resizeGlobalFieldList(FieldList<Dimension, DataType>& fieldList,
                             const DataType value,
                             const typename Field<Dimension, DataType>::FieldName name = "Unnamed Field",
                             const bool resetValues = true) const;
  template<typename DataType>
  void resizeFluidFieldList(FieldList<Dimension, DataType>& fieldList,
                            const DataType value,
                            const typename Field<Dimension, DataType>::FieldName name = "Unnamed Field",
                            const bool resetValues = true) const;
  template<typename DataType>
  void resizeSolidFieldList(FieldList<Dimension, DataType>& fieldList,
                            const DataType value,
                            const typename Field<Dimension, DataType>::FieldName name = "Unnamed Field",
                            const bool resetValues = true) const;

  //............................................................................
  // Create vector<vector<>> versions of the FieldLists.
  template<typename DataType> std::vector<std::vector<DataType>> newGlobalArray(const DataType value) const;
  template<typename DataType> std::vector<std::vector<DataType>> newFluidArray(const DataType value) const;
  template<typename DataType> std::vector<std::vector<DataType>> newSolidArray(const DataType value) const;

  // Resize vector<vector<>> versions of the FieldLists.
  template<typename DataType> void resizeGlobalArray(std::vector<std::vector<DataType>>& array,
                                                     const DataType value,
                                                     const bool resetValues = true) const;
  template<typename DataType> void resizeFluidArray(std::vector<std::vector<DataType>>& array,
                                                    const DataType value,
                                                    const bool resetValues = true) const;
  template<typename DataType> void resizeSolidArray(std::vector<std::vector<DataType>>& array,
                                                    const DataType value,
                                                    const bool resetValues = true) const;
  //............................................................................

  // Get the maximum kernel extent for all NodeLists.
  Scalar maxKernelExtent() const;

  // Compute coordinates bounding all nodes in the DataBase.
  void boundingBox(Vector& xmin, Vector& xmax,
                   const bool ghost = true) const;
  void boundingBox(Vector& xmin, Vector& xmax,
                   const FieldList<Dimension, int>& mask,
                   const bool ghost = true) const;

  // Return the local and global max sampling extents.
  void localSamplingBoundingVolume(Vector& centroid, double& radiusNodes, double& radiusSample,
                                   Vector& xminNodes, Vector& xmaxNodes,
                                   Vector& xminSample, Vector& xmaxSample) const;
  void globalSamplingBoundingVolume(Vector& centroid, double& radiusNodes, double& radiusSample,
                                    Vector& xminNodes, Vector& xmaxNodes,
                                    Vector& xminSample, Vector& xmaxSample) const;

  // Return the local min and max sampling extents for groupings of connected nodes.
  void localSamplingBoundingBoxes(std::vector<Vector>& xminima,
                                  std::vector<Vector>& xmaxima) const;
  void globalSamplingBoundingBoxes(std::vector<Vector>& xminima,
                                   std::vector<Vector>& xmaxima) const;

  // Provide a method to determine if the DataBase is in a minimally defined
  // valid state.
  bool valid() const;

private:
  //--------------------------- Private Interface ---------------------------//
  std::vector<NodeList<Dimension>*> mNodeListPtrs;

  std::vector<FluidNodeList<Dimension>*> mFluidNodeListPtrs;
  std::vector<NodeList<Dimension>*> mFluidNodeListAsNodeListPtrs;

  std::vector<SolidNodeList<Dimension>*> mSolidNodeListPtrs;
  std::vector<NodeList<Dimension>*> mSolidNodeListAsNodeListPtrs;

  mutable ConnectivityMapPtr mConnectivityMapPtr;

  // Prevent copying.
  DataBase(const DataBase& rhs);
};

}

#include "DataBaseInline.hh"

#else

namespace Spheral {
  template<typename Dimension> class DataBase;
}

#endif
