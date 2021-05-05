#include "Utilities/DBC.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Number of NodeLists registered with the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
unsigned int
DataBase<Dimension>::numNodeLists() const {
  return mNodeListPtrs.size();
}

//------------------------------------------------------------------------------
// Number of FluidNodeLists registered with the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
unsigned int
DataBase<Dimension>::numFluidNodeLists() const {
  return mFluidNodeListPtrs.size();
}

//------------------------------------------------------------------------------
// Number of SolidNodeLists registered with the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
int
DataBase<Dimension>::numSolidNodeLists() const {
  return mSolidNodeListPtrs.size();
}

//------------------------------------------------------------------------------
// Numbers of nodes in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
int
DataBase<Dimension>::numInternalNodes() const {
  int result = 0;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr != nodeListEnd();
       ++nodeListItr) result += (*nodeListItr)->numInternalNodes();
  return result;
}

template<typename Dimension>
inline
int
DataBase<Dimension>::numGhostNodes() const {
  int result = 0;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr != nodeListEnd();
       ++nodeListItr) result += (*nodeListItr)->numGhostNodes();
  return result;
}

template<typename Dimension>
inline
int
DataBase<Dimension>::numNodes() const {
  int result = 0;
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr != nodeListEnd();
       ++nodeListItr) result += (*nodeListItr)->numNodes();
  return result;
}

//------------------------------------------------------------------------------
// Numbers of fluid nodes in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
int
DataBase<Dimension>::numFluidInternalNodes() const {
  int result = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr) result += (*nodeListItr)->numInternalNodes();
  return result;
}

template<typename Dimension>
inline
int
DataBase<Dimension>::numFluidGhostNodes() const {
  int result = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr) result += (*nodeListItr)->numGhostNodes();
  return result;
}

template<typename Dimension>
inline
int
DataBase<Dimension>::numFluidNodes() const {
  int result = 0;
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr) result += (*nodeListItr)->numNodes();
  return result;
}

//------------------------------------------------------------------------------
// Standard STL like iterators for NodeLists.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
typename DataBase<Dimension>::NodeListIterator
DataBase<Dimension>::nodeListBegin() {
  return mNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::NodeListIterator
DataBase<Dimension>::nodeListEnd() {
  return mNodeListPtrs.end();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstNodeListIterator
DataBase<Dimension>::nodeListBegin() const {
  return mNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstNodeListIterator
DataBase<Dimension>::nodeListEnd() const {
  return mNodeListPtrs.end();
}

//------------------------------------------------------------------------------
// Standard STL like iterators for FluidNodeLists.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
typename DataBase<Dimension>::FluidNodeListIterator
DataBase<Dimension>::fluidNodeListBegin() {
  return mFluidNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::FluidNodeListIterator
DataBase<Dimension>::fluidNodeListEnd() {
  return mFluidNodeListPtrs.end();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstFluidNodeListIterator
DataBase<Dimension>::fluidNodeListBegin() const {
  return mFluidNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstFluidNodeListIterator
DataBase<Dimension>::fluidNodeListEnd() const {
  return mFluidNodeListPtrs.end();
}

//------------------------------------------------------------------------------
// Standard STL like iterators for FluidNodeLists, but over NodeList
// base type.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
typename DataBase<Dimension>::NodeListIterator
DataBase<Dimension>::fluidNodeListAsNodeListBegin() {
  return mFluidNodeListAsNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::NodeListIterator
DataBase<Dimension>::fluidNodeListAsNodeListEnd() {
  return mFluidNodeListAsNodeListPtrs.end();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstNodeListIterator
DataBase<Dimension>::fluidNodeListAsNodeListBegin() const {
  return mFluidNodeListAsNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstNodeListIterator
DataBase<Dimension>::fluidNodeListAsNodeListEnd() const {
  return mFluidNodeListAsNodeListPtrs.end();
}

//------------------------------------------------------------------------------
// Standard STL like iterators for SolidNodeLists.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
typename DataBase<Dimension>::SolidNodeListIterator
DataBase<Dimension>::solidNodeListBegin() {
  return mSolidNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::SolidNodeListIterator
DataBase<Dimension>::solidNodeListEnd() {
  return mSolidNodeListPtrs.end();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstSolidNodeListIterator
DataBase<Dimension>::solidNodeListBegin() const {
  return mSolidNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstSolidNodeListIterator
DataBase<Dimension>::solidNodeListEnd() const {
  return mSolidNodeListPtrs.end();
}

//------------------------------------------------------------------------------
// Standard STL like iterators for SolidNodeLists, but over NodeList
// base type.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
typename DataBase<Dimension>::NodeListIterator
DataBase<Dimension>::solidNodeListAsNodeListBegin() {
  return mSolidNodeListAsNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::NodeListIterator
DataBase<Dimension>::solidNodeListAsNodeListEnd() {
  return mSolidNodeListAsNodeListPtrs.end();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstNodeListIterator
DataBase<Dimension>::solidNodeListAsNodeListBegin() const {
  return mSolidNodeListAsNodeListPtrs.begin();
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConstNodeListIterator
DataBase<Dimension>::solidNodeListAsNodeListEnd() const {
  return mSolidNodeListAsNodeListPtrs.end();
}

//------------------------------------------------------------------------------
// Get the current connectivity map.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
const ConnectivityMap<Dimension>&
DataBase<Dimension>::
connectivityMap() const {
  VERIFY2(mConnectivityMapPtr.use_count() != 0,
          "DataBase::connectivityMap ERROR -- need to ensure ConnectivityMap is constructed before calling this method.");
  return *mConnectivityMapPtr;
}

template<typename Dimension>
inline
const ConnectivityMap<Dimension>&
DataBase<Dimension>::
connectivityMap(const bool computeGhostConnectivity,
                const bool computeOverlapConnectivity,
                const bool computeIntersectionConnectivity) const {
  if (mConnectivityMapPtr.use_count() == 0) this->updateConnectivityMap(computeGhostConnectivity, computeOverlapConnectivity, computeIntersectionConnectivity);
  return *mConnectivityMapPtr;
}

template<typename Dimension>
inline
typename DataBase<Dimension>::ConnectivityMapPtr
DataBase<Dimension>::
connectivityMapPtr(const bool computeGhostConnectivity,
                   const bool computeOverlapConnectivity,
                   const bool computeIntersectionConnectivity) const {
  if (mConnectivityMapPtr.use_count() == 0) this->updateConnectivityMap(computeGhostConnectivity, computeOverlapConnectivity, computeIntersectionConnectivity);
  return mConnectivityMapPtr;
}

//------------------------------------------------------------------------------
// Convenience method to construct a new FieldList with a Field for every
// NodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
FieldList<Dimension, DataType>
DataBase<Dimension>::
newGlobalFieldList(const DataType value,
                   const typename Field<Dimension, DataType>::FieldName name) const {
  FieldList<Dimension, DataType> result(FieldStorageType::CopyFields);
  for (ConstNodeListIterator nodeListItr = nodeListBegin();
       nodeListItr != nodeListEnd();
       ++nodeListItr) {
    result.appendNewField(name, **nodeListItr, value);
  }

  ENSURE(result.numFields() == numNodeLists());
  return result;
}

//------------------------------------------------------------------------------
// Convenience method to construct a new FieldList with a Field for every
// FluidNodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
FieldList<Dimension, DataType>
DataBase<Dimension>::
newFluidFieldList(const DataType value,
                  const typename Field<Dimension, DataType>::FieldName name) const {
  FieldList<Dimension, DataType> result(FieldStorageType::CopyFields);
  for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr) {
    result.appendNewField(name, **nodeListItr, value);
  }

  ENSURE(result.numFields() == numFluidNodeLists());
  return result;
}

//------------------------------------------------------------------------------
// Convenience method to construct a new FieldList with a Field for every
// SolidNodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
FieldList<Dimension, DataType>
DataBase<Dimension>::
newSolidFieldList(const DataType value,
                  const typename Field<Dimension, DataType>::FieldName name) const {
  FieldList<Dimension, DataType> result(FieldStorageType::CopyFields);
  for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
       nodeListItr != solidNodeListEnd();
       ++nodeListItr) {
    result.appendNewField(name, **nodeListItr, value);
  }

  ENSURE((int)result.numFields() == numSolidNodeLists());
  return result;
}

//------------------------------------------------------------------------------
// Convenience method to resize a FieldList such that it has a Field for every
// NodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
void
DataBase<Dimension>::
resizeGlobalFieldList(FieldList<Dimension, DataType>& fieldList,
                      const DataType value,
                      const typename Field<Dimension, DataType>::FieldName name,
                      const bool resetValues) const {
  VERIFY((fieldList.storageType() == FieldStorageType::CopyFields));

  // First check if it's necessary to resize the FieldList.
  bool reinitialize = fieldList.numFields() != numNodeLists();
  ConstNodeListIterator nodeListItr = nodeListBegin();
  typename FieldList<Dimension, DataType>::const_iterator itr = fieldList.begin();
  while (!reinitialize && 
         nodeListItr != nodeListEnd() &&
         itr != fieldList.end()) {
    reinitialize = (*itr)->nodeListPtr() != *nodeListItr;
    ++nodeListItr;
    ++itr;
  }

  if (reinitialize) {
    fieldList = FieldList<Dimension, DataType>(fieldList.storageType());
    for (ConstNodeListIterator nodeListItr = nodeListBegin();
         nodeListItr != nodeListEnd();
         ++nodeListItr) fieldList.appendNewField(name, **nodeListItr, value);
  } else if (resetValues) {
    fieldList = value;
  }

  ENSURE(fieldList.numFields() == numNodeLists());
}

//------------------------------------------------------------------------------
// Convenience method to resize a FieldList such that it has a Field for every
// FluidNodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
void
DataBase<Dimension>::
resizeFluidFieldList(FieldList<Dimension, DataType>& fieldList,
                     const DataType value,
                     const typename Field<Dimension, DataType>::FieldName name,
                     const bool resetValues) const {
  VERIFY((fieldList.storageType() == FieldStorageType::CopyFields));

  // First check if it's necessary to resize the FieldList.
  bool reinitialize = fieldList.numFields() != numFluidNodeLists();
  ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
  typename FieldList<Dimension, DataType>::const_iterator itr = fieldList.begin();
  while (!reinitialize && 
         nodeListItr != fluidNodeListEnd() &&
         itr != fieldList.end()) {
    reinitialize = (*itr)->nodeListPtr() != *nodeListItr;
    ++nodeListItr;
    ++itr;
  }

  if (reinitialize) {
    fieldList = FieldList<Dimension, DataType>(fieldList.storageType());
    for (ConstFluidNodeListIterator nodeListItr = fluidNodeListBegin();
         nodeListItr != fluidNodeListEnd();
         ++nodeListItr) fieldList.appendNewField(name, **nodeListItr, value);
  } else if (resetValues) {
    fieldList = value;
  }

  ENSURE(fieldList.numFields() == numFluidNodeLists());
}

//------------------------------------------------------------------------------
// Convenience method to resize a FieldList such that it has a Field for every
// SolidNodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
void
DataBase<Dimension>::
resizeSolidFieldList(FieldList<Dimension, DataType>& fieldList,
                     const DataType value,
                     const typename Field<Dimension, DataType>::FieldName name,
                     const bool resetValues) const {
  VERIFY((fieldList.storageType() == FieldStorageType::CopyFields));

  // First check if it's necessary to resize the FieldList.
  bool reinitialize = (int)fieldList.numFields() != numSolidNodeLists();
  ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
  typename FieldList<Dimension, DataType>::const_iterator itr = fieldList.begin();
  while (!reinitialize && 
         nodeListItr != solidNodeListEnd() &&
         itr != fieldList.end()) {
    reinitialize = (*itr)->nodeListPtr() != *nodeListItr;
    ++nodeListItr;
    ++itr;
  }

  if (reinitialize) {
    fieldList = FieldList<Dimension, DataType>(fieldList.storageType());
    for (ConstSolidNodeListIterator nodeListItr = solidNodeListBegin();
         nodeListItr != solidNodeListEnd();
         ++nodeListItr) fieldList.appendNewField(name, **nodeListItr, value);
  } else if (resetValues) {
    fieldList = value;
  }

  ENSURE((int)fieldList.numFields() == numSolidNodeLists());
}

//------------------------------------------------------------------------------
// Construct a new array<array> with an array for every NodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
std::vector<std::vector<DataType>>
DataBase<Dimension>::
newGlobalArray(const DataType value) const {
  std::vector<std::vector<DataType>> result;
  for (auto nodeListItr = nodeListBegin();
       nodeListItr != nodeListEnd();
       ++nodeListItr) {
    result.push_back(std::vector<DataType>((*nodeListItr)->numNodes(), value));
  }

  ENSURE(result.size() == numNodeLists());
  return result;
}

//------------------------------------------------------------------------------
// Construct a new array<array> with an array for every FluidNodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
std::vector<std::vector<DataType>>
DataBase<Dimension>::
newFluidArray(const DataType value) const {
  std::vector<std::vector<DataType>> result;
  for (auto nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr) {
    result.push_back(std::vector<DataType>((*nodeListItr)->numNodes(), value));
  }

  ENSURE(result.size() == numNodeLists());
  return result;
}

//------------------------------------------------------------------------------
// Construct a new array<array> with an array for every SolidNodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
std::vector<std::vector<DataType>>
DataBase<Dimension>::
newSolidArray(const DataType value) const {
  std::vector<std::vector<DataType>> result;
  for (auto nodeListItr = solidNodeListBegin();
       nodeListItr != solidNodeListEnd();
       ++nodeListItr) {
    result.push_back(std::vector<DataType>((*nodeListItr)->numNodes(), value));
  }

  ENSURE(result.size() == numNodeLists());
  return result;
}

//------------------------------------------------------------------------------
// Resize an array<array> for every NodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
void
DataBase<Dimension>::
resizeGlobalArray(std::vector<std::vector<DataType>>& array,
                  const DataType value,
                  const bool resetValues) const {
  array.resize(this->numNodeLists());
  auto k = 0;
  for (auto nodeListItr = nodeListBegin();
       nodeListItr != nodeListEnd();
       ++nodeListItr, ++k) {
    if (resetValues) {
      array[k] = std::vector<DataType>((*nodeListItr)->numNodes(), value);
    } else {
      array[k].resize((*nodeListItr)->numNodes(), value);
    }
  }

  ENSURE(array.size() == numNodeLists());
}

//------------------------------------------------------------------------------
// Resize an array<array> for every FluidNodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
void
DataBase<Dimension>::
resizeFluidArray(std::vector<std::vector<DataType>>& array,
                 const DataType value,
                 const bool resetValues) const {
  array.resize(this->numFluidNodeLists());
  auto k = 0;
  for (auto nodeListItr = fluidNodeListBegin();
       nodeListItr != fluidNodeListEnd();
       ++nodeListItr, ++k) {
    if (resetValues) {
      array[k] = std::vector<DataType>((*nodeListItr)->numNodes(), value);
    } else {
      array[k].resize((*nodeListItr)->numNodes(), value);
    }
  }

  ENSURE(array.size() == numNodeLists());
}

//------------------------------------------------------------------------------
// Resize an array<array> for every SolidNodeList in the DataBase.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename DataType>
inline
void
DataBase<Dimension>::
resizeSolidArray(std::vector<std::vector<DataType>>& array,
                 const DataType value,
                 const bool resetValues) const {
  array.resize(this->numSolidNodeLists());
  auto k = 0;
  for (auto nodeListItr = solidNodeListBegin();
       nodeListItr != solidNodeListEnd();
       ++nodeListItr, ++k) {
    if (resetValues) {
      array[k] = std::vector<DataType>((*nodeListItr)->numNodes(), value);
    } else {
      array[k].resize((*nodeListItr)->numNodes(), value);
    }
  }

  ENSURE(array.size() == numNodeLists());
}

}
