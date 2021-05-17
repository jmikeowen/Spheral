#include "NodePairList.hh"

namespace Spheral {
  
  NodePairList::NodePairList(){};

  void NodePairList::push_back(NodePairIdxType nodePair) {
    mDataArray.emplace_back(nodePair);
    //mDataArray.push_back(nodePair);
  } 

  void NodePairList::clear() {
    mDataArray.clear();
  }

}
