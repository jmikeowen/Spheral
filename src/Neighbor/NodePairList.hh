#ifndef _Spheral_NeighbourSpace_NodePairList_hh_
#define _Spheral_NeighbourSpace_NodePairList_hh_

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

//------------------------------------------------------------------------------
struct NodePairIdxType {
  NodePairIdxType(int i_n, int i_l, int j_n, int j_l,
                  double f = 1.0);
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
  bool operator< (const NodePairIdxType& val) const { return (this->hash() <  val.hash()); }
};

//------------------------------------------------------------------------------
class NodePairList {
public:
  typedef std::vector<NodePairIdxType> ContainerType;
  typedef typename ContainerType::value_type value_type;
  typedef typename ContainerType::reference reference;
  typedef typename ContainerType::const_reference const_reference;
  typedef typename ContainerType::iterator iterator;
  typedef typename ContainerType::const_iterator const_iterator;
  typedef typename ContainerType::reverse_iterator reverse_iterator;
  typedef typename ContainerType::const_reverse_iterator const_reverse_iterator;

  NodePairList();
  void push_back(NodePairIdxType nodePair);
  void clear(); 
  size_t size() const { return mNodePairList.size(); }

  // Iterators
  iterator begin() { return mNodePairList.begin(); }
  iterator end() { return mNodePairList.end(); }
  const_iterator begin() const { return mNodePairList.begin(); }
  const_iterator end() const { return mNodePairList.end(); }

  // Reverse iterators
  reverse_iterator rbegin() { return mNodePairList.rbegin(); }
  reverse_iterator rend() { return mNodePairList.rend(); }
  const_reverse_iterator rbegin() const { return mNodePairList.rbegin(); }
  const_reverse_iterator rend() const { return mNodePairList.rend(); }

  // Indexing
  reference operator[](const size_t i) { return mNodePairList[i]; }
  const_reference operator[](const size_t i) const { return mNodePairList[i]; }

  // Inserting
  template<typename InputIterator>
  iterator insert(const_iterator pos, InputIterator first, InputIterator last) { return mNodePairList.insert(pos, first, last); }

private:
  ContainerType mNodePairList;
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
