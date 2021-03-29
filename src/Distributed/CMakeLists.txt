include_directories(.)
set(Distributed_sources Communicator.cc)

if (ENABLE_1D)
  list(APPEND Distributed_sources SortAndDivideRedistributeNodes1d.cc)
endif()

if (ENABLE_2D)
  list(APPEND Distributed_sources SortAndDivideRedistributeNodes2d.cc)
endif()

if (ENABLE_3D)
  list(APPEND Distributed_sources SortAndDivideRedistributeNodes3d.cc)
endif()

if (ENABLE_MPI)
   list(APPEND Distributed_sources waitAllWithDeadlockDetection.cc)
endif()

set(Distributed_inst
    DistributedBoundary
    NestedGridDistributedBoundary
    TreeDistributedBoundary
    BoundingVolumeDistributedBoundary
    RedistributeNodes
    DistributeByXPosition
    SortAndDivideRedistributeNodes
    SpaceFillingCurveRedistributeNodes
    MortonOrderRedistributeNodes
    PeanoHilbertOrderRedistributeNodes
    VoronoiRedistributeNodes
    )

instantiate(Distributed_inst Distributed_sources)

set(Distributed_headers
    Communicator.hh
    CompareDomainNodesByPosition.hh
    DistributedBoundary.hh
    DistributedBoundaryInline.hh
    #DomainNode.hh
    #DomainNodeInline.hh
    NestedGridDistributedBoundary.hh
    NestedGridUtilities.hh
    RedistributeNodes.hh
    RedistributeNodesInline.hh
    TreeDistributedBoundary.hh
    waitAllWithDeadlockDetection.hh
    SortAndDivideRedistributeNodes.hh
    SortAndDivideRedistributeNodes1d.hh
    )

spheral_add_cxx_library(Distributed)

