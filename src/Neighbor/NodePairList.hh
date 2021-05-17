#ifndef _Spheral_NeighbourSpace_NodePairList_hh_
#define _Spheral_NeighbourSpace_NodePairList_hh_

#include "LvArray/Array.hpp"
#include "LvArray/ChaiBuffer.hpp"

#include "Utilities/size_t_bits.hh"
#include "Utilities/DBC.hh"

#include <iostream>
#include <vector>
#include <tuple>
#include <functional>
#include <iostream>
// #include <boost/container_hash/hash.hpp>

// These are based on what we get from size_t_bits
#define MAX_NODE_INDEX (size_t(1u) << ((SIZE_T_BITS - 10)/2))
#define MAX_NODELIST_INDEX (size_t(1u) << 5)

namespace Spheral {

namespace detail{
  template<typename T>  class DeviceAccessor;
}

//------------------------------------------------------------------------------
struct NodePairIdxType {
  RAJA_HOST_DEVICE NodePairIdxType(int i_n = 0, int i_l = 0,
                  int j_n = 0, int j_l = 0,
                  double f = 1.0) : i_node(i_n), i_list(i_l), j_node(j_n), j_list(j_l), f_couple(f) {}
  int i_node, i_list, j_node, j_list;
  double f_couple;                       // An arbitrary fraction in [0,1] to hold the effective coupling of the pair

  size_t hash() const {
    // We do this with simple bit shifting, requiring max values for the integer
    // components.  We assume the
    //    i_list, j_list < 32 (2^5)
    //    i_node, j_node < 134217728 (2^27) (on 64 bit machines)
    REQUIRE(size_t(i_node) < MAX_NODE_INDEX);
    REQUIRE(size_t(j_node) < MAX_NODE_INDEX);
    REQUIRE(size_t(i_list) < MAX_NODELIST_INDEX);
    REQUIRE(size_t(j_list) < MAX_NODELIST_INDEX);
    return ((size_t(i_list) << (SIZE_T_BITS - 5)) +
            (size_t(i_node) << (SIZE_T_BITS/2)) +
            (size_t(j_list) << (SIZE_T_BITS/2 - 5)) +
            size_t(j_node));
  }

  // Comparisons
  bool operator==(const NodePairIdxType& val) const { return (this->hash() == val.hash()); }
  bool operator!=(const NodePairIdxType& val) const { return !(this->hash() == val.hash()); }
  bool operator< (const NodePairIdxType& val) const { return (this->hash() <  val.hash()); }
};

//------------------------------------------------------------------------------
class NodePairList {
public:
  using ValueType = NodePairIdxType;
  typedef ValueType& reference;
  typedef const ValueType& const_reference;

  using ContainerType = LvArray::Array< ValueType, 1, camp::idx_seq<0>, std::ptrdiff_t, LvArray::ChaiBuffer >;
  using ContainerTypeView = LvArray::ArrayView< ValueType, 1, 0, std::ptrdiff_t, LvArray::ChaiBuffer >;

  typedef NodePairIdxType* iterator;
  typedef const NodePairIdxType* const_iterator;

  //typedef std::vector<ValueType> ContainerType;
  ////typedef typename ContainerType::reference reference;
  ////typedef typename ContainerType::const_reference const_reference;
  //typedef typename ContainerType::iterator iterator;
  //typedef typename ContainerType::const_iterator const_iterator;
  ////typedef typename ContainerType::reverse_iterator reverse_iterator;
  ////typedef typename ContainerType::const_reverse_iterator const_reverse_iterator;

  NodePairList();
  void push_back(NodePairIdxType nodePair);
  void clear(); 
  size_t size() const { return mDataArray.size(); }

  //// Iterators
  //iterator begin() { return mDataArray.begin(); }
  //iterator end() { return mDataArray.end(); }
  iterator begin() { return &mDataArray[0]; }
  iterator end() { return &mDataArray[size()-1]; }
  //const_iterator begin() const { return mDataArray.begin(); }
  //const_iterator end() const { return mDataArray.end(); }

  //// Reverse iterators
  //reverse_iterator rbegin() { return mDataArray.rbegin(); }
  //reverse_iterator rend() { return mDataArray.rend(); }
  //const_reverse_iterator rbegin() const { return mDataArray.rbegin(); }
  //const_reverse_iterator rend() const { return mDataArray.rend(); }

  //// Indexing
  reference operator[](const size_t i) { return mDataArray[i]; }
  const_reference operator[](const size_t i) const { return mDataArray[i]; }

  // Inserting
  template<typename InputIterator>
  void insert(size_t pos, InputIterator first, InputIterator last) { mDataArray.insert(pos, first, last); }
  //iterator insert(const_iterator pos, InputIterator first, InputIterator last) { return mDataArray.insert(pos, first, last); }

private:
  ContainerType mDataArray;

  friend class detail::DeviceAccessor<NodePairList>;
};


//------------------------------------------------------------------------------
// Output for NodePairIdxType
inline
std::ostream& operator<<(std::ostream& os, const NodePairIdxType& x) {
  os << "[(" << x.i_list << " " << x.i_node << ") (" << x.j_list << " " << x.j_node << ")]";
  return os;
}

} //namespace Spheral

//------------------------------------------------------------------------------
// Provide a method of hashing NodePairIdxType
namespace std {
  template<>
  struct hash<Spheral::NodePairIdxType> {
    size_t operator()(const Spheral::NodePairIdxType& x) const {
      return x.hash();
      // boost::hash<std::tuple<int, int, int, int>> hasher;
      // return hasher(std::make_tuple(x.i_node, x.i_list, x.j_node, x.j_list));
    }
  };
} // namespace std


#endif // _Spheral_NeighbourSpace_NodePairList_hh_
