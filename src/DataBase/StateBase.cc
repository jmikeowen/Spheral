//---------------------------------Spheral++----------------------------------//
// StateBase -- The base class for State and StateDerivatives method, providing
// the common methods for registering and storing Fields, as well as serving
// up FieldLists based on these Fields.
//
// Created by JMO, Wed Aug 25 22:23:35 2004
//----------------------------------------------------------------------------//
#include "StateBase.hh"
#include "Field/Field.hh"
#include "Field/FieldList.hh"
#include "Neighbor/ConnectivityMap.hh"
#include "Mesh/Mesh.hh"
#include "Utilities/DBC.hh"

#include <algorithm>
#include <sstream>
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::abs;

namespace Spheral {

// namespace {
// //------------------------------------------------------------------------------
// // Helper for copying a type, used in copyState
// //------------------------------------------------------------------------------
// template<typename T>
// T*
// extractType(boost::any& anyT) {
//   try {
//     T* result = boost::any_cast<T*>(anyT);
//     return result;
//   } catch (boost::any_cast_error) {
//     return NULL;
//   }
// }

// }

//------------------------------------------------------------------------------
// Default constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
StateBase<Dimension>::
StateBase():
  mStorage(),
  mCache(),
  mConnectivityMapPtr(),
  mMeshPtr(new MeshType()) {
}

//------------------------------------------------------------------------------
// Copy constructor.
//------------------------------------------------------------------------------
template<typename Dimension>
StateBase<Dimension>::
StateBase(const StateBase<Dimension>& rhs):
  mStorage(rhs.mStorage),
  mCache(),
  mNodeListPtrs(rhs.mNodeListPtrs),
  mConnectivityMapPtr(rhs.mConnectivityMapPtr),
  mMeshPtr(rhs.mMeshPtr) {
}

//------------------------------------------------------------------------------
// Destructor.
//------------------------------------------------------------------------------
template<typename Dimension>
StateBase<Dimension>::
~StateBase() {
}

//------------------------------------------------------------------------------
// Assignment.
//------------------------------------------------------------------------------
template<typename Dimension>
StateBase<Dimension>&
StateBase<Dimension>::
operator=(const StateBase<Dimension>& rhs) {
  if (this != &rhs) {
    mStorage = rhs.mStorage;
    mCache = CacheType();
    mNodeListPtrs = rhs.mNodeListPtrs;
    mConnectivityMapPtr = rhs.mConnectivityMapPtr;
    mMeshPtr = rhs.mMeshPtr;
  }
  return *this;
}

//------------------------------------------------------------------------------
// Test if the internal state is equal.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
StateBase<Dimension>::
operator==(const StateBase<Dimension>& rhs) const {
  if (mStorage.size() != rhs.mStorage.size()) {
    cerr << "Storage sizes don't match." << endl;
    return false;
  }
  vector<KeyType> lhsKeys = keys();
  vector<KeyType> rhsKeys = rhs.keys();
  if (lhsKeys.size() != rhsKeys.size()) {
    cerr << "Keys sizes don't match." << endl;
    return false;
  }
  sort(lhsKeys.begin(), lhsKeys.end());
  sort(rhsKeys.begin(), rhsKeys.end());
  if (lhsKeys != rhsKeys) {
    cerr << "Keys don't match." << endl;
    return false;
  }

  // Walk the keys, and rely on the virtual overloaded 
  // Field::operator==(FieldBase) to do the right thing!
  // We are also relying here on the fact that std::map with a given
  // set of keys will always result in the same order.
  bool result = true;
  typename StorageType::const_iterator lhsItr, rhsItr;
  for (rhsItr = rhs.mStorage.begin(), lhsItr = mStorage.begin();
       rhsItr != rhs.mStorage.end();
       ++rhsItr, ++lhsItr) {
    try {
      auto lhsPtr = boost::any_cast<FieldBase<Dimension>*>(lhsItr->second);
      auto rhsPtr = boost::any_cast<FieldBase<Dimension>*>(rhsItr->second);
      if (*lhsPtr != *rhsPtr) {
        cerr << "Fields for " << lhsItr->first <<  " don't match." << endl;
        result = false;
      }
    } catch (const boost::bad_any_cast&) {
      try {
        auto lhsPtr = boost::any_cast<vector<Vector>*>(lhsItr->second);
        auto rhsPtr = boost::any_cast<vector<Vector>*>(rhsItr->second);
        if (*lhsPtr != *rhsPtr) {
          cerr << "vector<Vector> for " << lhsItr->first <<  " don't match." << endl;
          result = false;
        }
      } catch (const boost::bad_any_cast&) {
        std::cerr << "StateBase::operator== WARNING: unable to compare values for " << lhsItr->first << "\n";
      }
    }
  }
  return result;
}

//------------------------------------------------------------------------------
// Test if the given key is registered.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
StateBase<Dimension>::
registered(const StateBase<Dimension>::KeyType& key) const {
  return (mStorage.find(key) != mStorage.end());
}

//------------------------------------------------------------------------------
// Test if the given field is registered.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
StateBase<Dimension>::
registered(const FieldBase<Dimension>& field) const {
  const KeyType key = this->key(field);
  typename StorageType::const_iterator itr = mStorage.find(key);
  return (itr != mStorage.end());
}

//------------------------------------------------------------------------------
// Test if the given FieldList is registered.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
StateBase<Dimension>::
registered(const FieldListBase<Dimension>& fieldList) const {
  REQUIRE(fieldList.begin_base() != fieldList.end_base());
  return this->registered(**fieldList.begin_base());
}

//------------------------------------------------------------------------------
// Test if the given field name is registered.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
StateBase<Dimension>::
fieldNameRegistered(const FieldName& name) const {
  KeyType fieldName, nodeListName;
  auto itr = mStorage.begin();
  while (itr != mStorage.end()) {
    splitFieldKey(itr->first, fieldName, nodeListName);
    if (fieldName == name) return true;
    ++itr;
  }
  return false;
}

//------------------------------------------------------------------------------
// Enroll a field.
//------------------------------------------------------------------------------
template<typename Dimension>
void
StateBase<Dimension>::
enroll(FieldBase<Dimension>& field) {
  const KeyType key = this->key(field);
  boost::any fieldptr;
  fieldptr = &field;
  mStorage[key] = fieldptr;
  mNodeListPtrs.insert(field.nodeListPtr());
  // std::cerr << "StateBase::enroll field:  " << key << " at " << &field << std::endl;
  ENSURE(&(this->getAny<FieldBase<Dimension>>(key)) == &field);
  ENSURE(find(mNodeListPtrs.begin(), mNodeListPtrs.end(), field.nodeListPtr()) != mNodeListPtrs.end());
}

//------------------------------------------------------------------------------
// Enroll a field (shared_ptr).
//------------------------------------------------------------------------------
template<typename Dimension>
void
StateBase<Dimension>::
enroll(std::shared_ptr<FieldBase<Dimension>>& fieldPtr) {
  const KeyType key = this->key(*fieldPtr);
  mStorage[key] = fieldPtr.get();
  mNodeListPtrs.insert(fieldPtr->nodeListPtr());
  mFieldCache.push_back(fieldPtr);
  ENSURE(find(mNodeListPtrs.begin(), mNodeListPtrs.end(), fieldPtr->nodeListPtr()) != mNodeListPtrs.end());
}

//------------------------------------------------------------------------------
// Add the fields from a FieldList.
//------------------------------------------------------------------------------
template<typename Dimension>
void
StateBase<Dimension>::
enroll(FieldListBase<Dimension>& fieldList) {
  for (auto itr = fieldList.begin_base();
       itr != fieldList.end_base();
       ++itr) {
    this->enroll(**itr);
  }
}

//------------------------------------------------------------------------------
// Return the full set of known keys.
//------------------------------------------------------------------------------
template<typename Dimension>
std::vector<typename StateBase<Dimension>::KeyType>
StateBase<Dimension>::
keys() const {
  vector<KeyType> result;
  result.reserve(mStorage.size());
  for (auto itr = mStorage.begin();
       itr != mStorage.end();
       ++itr) result.push_back(itr->first);
  ENSURE(result.size() == mStorage.size());
  return result;
}

//------------------------------------------------------------------------------
// Return the set of registered Field names.
//------------------------------------------------------------------------------
template<typename Dimension>
std::vector<typename FieldBase<Dimension>::FieldName>
StateBase<Dimension>::
fieldKeys() const {
  KeyType fieldName, nodeListName;
  vector<typename FieldBase<Dimension>::FieldName> result;
  result.reserve(mStorage.size());
  for (typename StorageType::const_iterator itr = mStorage.begin();
       itr != mStorage.end();
       ++itr) {
    splitFieldKey(itr->first, fieldName, nodeListName);
    if (fieldName != "" and nodeListName != "") result.push_back(fieldName);
  }

  // Remove any duplicates.  This will happen when we've stored the same field
  // for different NodeLists.
  sort(result.begin(), result.end());
  result.erase(unique(result.begin(), result.end()), result.end());

  return result;
}

//------------------------------------------------------------------------------
// Enroll a ConnectivityMap.
//------------------------------------------------------------------------------
template<typename Dimension>
void
StateBase<Dimension>::
enrollConnectivityMap(typename StateBase<Dimension>::ConnectivityMapPtr connectivityMapPtr) {
  mConnectivityMapPtr = connectivityMapPtr;
}

//------------------------------------------------------------------------------
// Return the ConnectivityMap (non-const version)
//------------------------------------------------------------------------------
template<typename Dimension>
typename StateBase<Dimension>::ConnectivityMapType&
StateBase<Dimension>::
connectivityMap() {
  REQUIRE(mConnectivityMapPtr.use_count() != 0);
  return *mConnectivityMapPtr;
}

//------------------------------------------------------------------------------
// Return the ConnectivityMap (const version)
//------------------------------------------------------------------------------
template<typename Dimension>
const typename StateBase<Dimension>::ConnectivityMapType&
StateBase<Dimension>::
connectivityMap() const {
  REQUIRE(mConnectivityMapPtr.use_count() != 0);
  return *mConnectivityMapPtr;
}

//------------------------------------------------------------------------------
// Enroll an external mesh.
//------------------------------------------------------------------------------
template<typename Dimension>
void
StateBase<Dimension>::
enrollMesh(typename StateBase<Dimension>::MeshPtr meshPtr) {
  mMeshPtr = meshPtr;
}

//------------------------------------------------------------------------------
// Test if a mesh is currently available.
//------------------------------------------------------------------------------
template<typename Dimension>
bool
StateBase<Dimension>::
meshRegistered() const {
  return (mMeshPtr.use_count() != 0);
}

//------------------------------------------------------------------------------
// Return the mesh. (non-const version)
//------------------------------------------------------------------------------
template<typename Dimension>
Mesh<Dimension>&
StateBase<Dimension>::
mesh() {
  REQUIRE(meshRegistered());
  return *mMeshPtr;
}

//------------------------------------------------------------------------------
// Return the mesh. (const version)
//------------------------------------------------------------------------------
template<typename Dimension>
const Mesh<Dimension>&
StateBase<Dimension>::
mesh() const {
  REQUIRE(meshRegistered());
  return *mMeshPtr;
}

//------------------------------------------------------------------------------
// Assign the state data of this State object to be equal to the values
// in another.
//------------------------------------------------------------------------------
template<typename Dimension>
void
StateBase<Dimension>::
assign(const StateBase<Dimension>& rhs) {

  // Extract the keys for each state, and verify they line up.
  REQUIRE(mStorage.size() == rhs.mStorage.size());
  vector<KeyType> lhsKeys = keys();
  vector<KeyType> rhsKeys = rhs.keys();
  REQUIRE(lhsKeys.size() == rhsKeys.size());
  sort(lhsKeys.begin(), lhsKeys.end());
  sort(rhsKeys.begin(), rhsKeys.end());
  REQUIRE(lhsKeys == rhsKeys);

  // Walk the keys, and rely on the underlying type to know how to copy itself.
  for (typename StorageType::const_iterator itr = rhs.mStorage.begin();
       itr != rhs.mStorage.end();
       ++itr) {
    auto& anylhs = mStorage[itr->first];
    const auto& anyrhs = itr->second;
    try {
      auto lhsptr = boost::any_cast<FieldBase<Dimension>*>(anylhs);
      const auto rhsptr = boost::any_cast<FieldBase<Dimension>*>(anyrhs);
      *lhsptr = *rhsptr;
    } catch(const boost::bad_any_cast&) {
      try {
        auto lhsptr = boost::any_cast<vector<Vector>*>(anylhs);
        const auto rhsptr = boost::any_cast<vector<Vector>*>(anyrhs);
        *lhsptr = *rhsptr;
      } catch(const boost::bad_any_cast&) {
        // We'll assume other things don't need to be assigned...
        // VERIFY2(false, "StateBase::assign ERROR: unknown type for key " << itr->first << "\n");
      }
    }
  }

  // Copy the connectivity (by reference).  This thing is too
  // big to carry around separate copies!
  if (rhs.mConnectivityMapPtr != NULL) {
    mConnectivityMapPtr = rhs.mConnectivityMapPtr;
  } else {
    mConnectivityMapPtr = ConnectivityMapPtr();
  }

  // Copy the mesh.
  if (rhs.mMeshPtr != NULL) {
    mMeshPtr = MeshPtr(new MeshType());
    *mMeshPtr = *(rhs.mMeshPtr);
  } else {
    mMeshPtr = MeshPtr();
  }
}

//------------------------------------------------------------------------------
// Force the state fields to be copied to local storage.
//------------------------------------------------------------------------------
template<typename Dimension>
void
StateBase<Dimension>::
copyState() {

  // Remove any pre-existing stuff.
  mCache = CacheType();
  mFieldCache = FieldCacheType();

  // Walk the registered state and copy it to our local cache.
  for (auto itr = mStorage.begin();
       itr != mStorage.end();
       ++itr) {
    boost::any anythingPtr = itr->second;

    // Is this a Field?
    try {
      auto ptr = boost::any_cast<FieldBase<Dimension>*>(anythingPtr);
      mFieldCache.push_back(ptr->clone());
      itr->second = mFieldCache.back().get();

    } catch (const boost::bad_any_cast&) {
      try {
        auto ptr = boost::any_cast<vector<Vector>*>(anythingPtr);
        auto clone = std::shared_ptr<vector<Vector>>(new vector<Vector>(*ptr));
        mCache.push_back(clone);
        itr->second = clone.get();

      } catch (const boost::bad_any_cast&) {
        // We'll assume other things don't need to be copied...
        // VERIFY2(false, "StateBase::copyState ERROR: unrecognized type for " << itr->first << "\n");
      }
    }
  }
}

}

