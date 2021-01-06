include_directories(.)
set(Neighbor_inst
    ConnectivityMap
    GridCellIndex
    GridCellPlane
    Neighbor
    NestedGridNeighbor
    TreeNeighbor
   )


set(Neighbor_sources 
    NodePairList.cc)

instantiate(Neighbor_inst Neighbor_sources)

set(Neighbor_headers
    ConnectivityMap.hh
    ConnectivityMapInline.hh
    GridCellIndex.hh
    GridCellIndexBase.hh
    GridCellIndexInline.hh
    GridCellPlane.hh
    GridCellPlaneInline.hh
    Neighbor.hh
    NeighborInline.hh
    NestedGridNeighbor.hh
    NestedGridNeighborInline.hh
    TreeNeighbor.hh
    NodePairList.hh
    )

spheral_add_cxx_library(Neighbor)

list(APPEND SPHERAL_CXX_LIBS Spheral_Neighbor)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
