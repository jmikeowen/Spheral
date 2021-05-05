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
    NodePairList.hh
    TreeNeighbor.hh
    )

spheral_add_cxx_library(Neighbor)

