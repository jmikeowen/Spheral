#include "boost/algorithm/string.hpp"
#include "DataBase/UpdatePolicyBase.hh"
#include "Mesh/Mesh.hh"
#include "Utilities/DBC.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Return the Field for the given key.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename Value>
Field<Dimension, Value>&
StateBase<Dimension>::
field(const typename StateBase<Dimension>::KeyType& key, 
      const Value&) const {
  try {
    return dynamic_cast<Field<Dimension, Value>&>(this->getAny<FieldBase<Dimension>>(key));
  } catch (...) {
    VERIFY2(false,"StateBase ERROR: unable to extract field for key " << key << "\n");
  }
}

//------------------------------------------------------------------------------
// Return all the Fields of the given Value element type.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename Value>
inline
std::vector<Field<Dimension, Value>*>
StateBase<Dimension>::
allFields(const Value&) const {
  std::vector<Field<Dimension, Value>*> result;
  KeyType fieldName, nodeListName;
  for (auto itr = mStorage.begin();
       itr != mStorage.end();
       ++itr) {
    try {
      Field<Dimension, Value>* ptr = dynamic_cast<Field<Dimension, Value>*>(boost::any_cast<FieldBase<Dimension>*>(itr->second));
      if (ptr != 0) result.push_back(ptr);
    } catch (...) {
      // The field must have been the wrong type.
    }
  }
  return result;
}

//------------------------------------------------------------------------------
// Return a FieldList containing all registered fields of the given name.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename Value>
inline
FieldList<Dimension, Value>
StateBase<Dimension>::
//fields(const SPHString& name, const Value& dummy) const {
fields(const std::string& name, const Value& dummy) const {
  FieldList<Dimension, Value> result;
  KeyType fieldName, nodeListName;
  for (auto itr = mStorage.begin();
       itr != mStorage.end();
       ++itr) {
    splitFieldKey(itr->first, fieldName, nodeListName);
    if (fieldName == name) {
      CHECK(nodeListName != "");
      result.appendField(this->field<Value>(itr->first, dummy));
    }
  }
  return result;
}

//------------------------------------------------------------------------------
// Enroll an arbitrary type
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename Value>
void
StateBase<Dimension>::
enrollAny(const typename StateBase<Dimension>::KeyType& key, Value& thing) {
  mStorage[key] = &thing;
}

//------------------------------------------------------------------------------
// Extract an arbitrary type
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename Value>
Value&
StateBase<Dimension>::
getAny(const typename StateBase<Dimension>::KeyType& key) const {
  try {
    Value& result = *boost::any_cast<Value*>(mStorage.find(key)->second);
    return result;
  } catch (const boost::bad_any_cast&) {
    VERIFY2(false, "StateBase::getAny ERROR: unable to extract Value for " << key << "\n");
  }
}

// Same thing passing a dummy argument to help with template type
template<typename Dimension>
template<typename Value>
Value&
StateBase<Dimension>::
getAny(const typename StateBase<Dimension>::KeyType& key,
       const Value&) const {
  return this->getAny<Value>(key);
}

//------------------------------------------------------------------------------
// Construct the lookup key for the given field.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
typename StateBase<Dimension>::KeyType
StateBase<Dimension>::
key(const FieldBase<Dimension>& field) {
  return buildFieldKey(field.name(), field.nodeListPtr()->name());
}

//------------------------------------------------------------------------------
// Construct the lookup key for the given FieldList.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
typename StateBase<Dimension>::KeyType
StateBase<Dimension>::
key(const FieldListBase<Dimension>& fieldList) {
  REQUIRE(fieldList.begin_base() != fieldList.end_base());
  return buildFieldKey((*fieldList.begin_base())->name(), UpdatePolicyBase<Dimension>::wildcard());
}

//------------------------------------------------------------------------------
// Assign the Fields matching the given name of this State object to be equal to
// the values in another.
//------------------------------------------------------------------------------
template<typename Dimension>
template<typename Value>
inline
void
StateBase<Dimension>::
assignFields(const StateBase<Dimension>& rhs, const std::string name) {
  auto lhsfields = this->fields(name, Value());
  auto rhsfields = rhs.fields(name, Value());
  lhsfields.assignFields(rhsfields);
}

//------------------------------------------------------------------------------
// Internal methods to encode the convention for combining Field and NodeList
// names into a single unique key.
//------------------------------------------------------------------------------
template<typename Dimension>
inline
typename StateBase<Dimension>::KeyType
StateBase<Dimension>::
//buildFieldKey(const SPHString& fieldName,
              //const SPHString& nodeListName) {
buildFieldKey(const std::string& fieldName,
              const std::string& nodeListName) {
  //return fieldName + SPHString("|") + nodeListName;
  return fieldName + "|" + nodeListName;
}

template<typename Dimension>
inline
void
StateBase<Dimension>::
splitFieldKey(const KeyType& key,
              KeyType& fieldName,
              KeyType& nodeListName) {
  std::vector<std::string> keys;
  boost::split(keys, key, boost::is_any_of("|"));
  if (keys.size() > 1) {
    fieldName = keys[0];
    nodeListName = keys[1];
  } else if (keys.size() == 1) {
    fieldName = keys[0];
    nodeListName = "";
  } else {
    fieldName = "";
    nodeListName = "";
  }
}

}
