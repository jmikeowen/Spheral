from PYB11Generator import *

#-------------------------------------------------------------------------------
# NodePairIdxType
#-------------------------------------------------------------------------------
class NodePairIdxType:
    def pyinit(self,
               i_n = "int",
               i_l = "int",
               j_n = "int",
               j_l = "int",
               f = ("double", "1.0")):
      "Constructor"
      return

    # Comparisons
    def __eq__(self):
      return
    def __lt__(self):
      return

    # String representation
    @PYB11implementation("""
[](const NodePairIdxType& self) {
  std::string result = "(" + 
                       std::to_string(self.i_list) + " " +
                       std::to_string(self.i_node) + " " +
                       std::to_string(self.j_list) + " " +
                       std::to_string(self.j_node) + ")";
  return result;
}""")
    def __repr__(self):
        return

    # Attributes
    i_node = PYB11readwrite()
    j_node = PYB11readwrite()
    i_list = PYB11readwrite()
    j_list = PYB11readwrite()
    f_couple = PYB11readwrite()

#-------------------------------------------------------------------------------
# NodePairList
#-------------------------------------------------------------------------------
class NodePairList:
  def pyinit(self):
    "Default Constructor"

  def push_back(self,
                nodePair = "NodePairIdxType"):
    "Push new Node Idx Data onto vector"
    return "void"

  def clear(self):
    "Clears all data from Node pair List."
    return "void"

  def size(self):
    "Returns the number of Node Pairs in the lsit"
    return "size_t"

  @PYB11implementation("[](const NodePairList& self) { return self.size(); }")
  def __len__(self):
    return

  @PYB11cppname("operator[]")
  @PYB11returnpolicy("reference_internal")
  @PYB11implementation('[](NodePairList& self, int i) { const int n = self.size(); return &self[(i %% n + n) %% n]; }')
  def __getitem__(self):
      return

  #@PYB11implementation("[](const NodePairList& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0,1>()")
  #def __iter__(self):
  #  "Python iteration through a NodePairList."

  @PYB11implementation("""[](const NodePairList& self, const NodePairIdxType val) { for (const auto& ele: self) {
                                                                                      if (ele == val) return true;
                                                                                    }
                                                                                    return false;
                                                                                  }""")
  def __contains__(self,
                   val = "NodePairIdxType"):
    return "bool"

  @PYB11implementation("""[](const NodePairList& self, const py::tuple val0) { if (val0.size() != 4) throw py::value_error("require a tuple of 4 integers");
                                                                               NodePairIdxType val(val0[0].cast<int>(), val0[1].cast<int>(), val0[2].cast<int>(), val0[3].cast<int>());
                                                                               //for (const auto& ele: self) {
                                                                               //  if (ele == val) return true;
                                                                               for (size_t i = 0; i < self.size(); i++) {
                                                                                 if (self[i] == val) return true;
                                                                               }
                                                                               return false;
                                                                             }""")
  def __contains__(self,
                   val = "py::tuple"):
    return "bool"

