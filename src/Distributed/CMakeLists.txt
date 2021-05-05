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
    BoundingVolumeDistributedBoundary.hh
    Communicator.hh
    CompareDomainNodesByPosition.hh
    DistributeByXPosition.hh
    DistributedBoundary.hh
    DistributedBoundaryInline.hh
    MortonOrderRedistributeNodes.hh
    NestedGridDistributedBoundary.hh
    NestedGridRedistributeNodes.hh
    NestedGridRedistributeNodesInline.hh
    NestedGridUtilities.hh
    ParmetisRedistributeNodes.hh
    ParmetisRedistributeNodesInline.hh
    PeanoHilbertOrderRedistributeNodes.hh
    RedistributeNodes.hh
    RedistributeNodesInline.hh
    SortAndDivideRedistributeNodes.hh
    SortAndDivideRedistributeNodes1d.hh
    SortAndDivideRedistributeNodes2d.hh
    SortAndDivideRedistributeNodes3d.hh
    SortAndDivideRedistributeNodesInline.hh
    SpaceFillingCurveRedistributeNodes.hh
    TreeDistributedBoundary.hh
    VoronoiRedistributeNodes.hh
    waitAllWithDeadlockDetection.hh
    )

spheral_add_cxx_library(Distributed)

