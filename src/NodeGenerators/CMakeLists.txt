include_directories(.)
set(NodeGenerators_inst
    relaxNodeDistribution
    centroidalRelaxNodesImpl
    compactFacetedVolumes
    chooseRandomNonoverlappingCenter
   )

set(NodeGenerators_sources
    generateCylDistributionFromRZ.cc
    fillFacetedVolume.cc
    readSiloPolyMesh.cc
   )

instantiate(NodeGenerators_inst NodeGenerators_sources)

set(NodeGenerators_headers
    fillFacetedVolume.hh
    generateCylDistributionFromRZ.hh
    )

spheral_install_python_files(
  NodeGeneratorBase.py
  GenerateNodeDistribution2d.py
  GenerateNodeDistribution3d.py
  CompositeNodeDistribution.py
  distributeNodesGeneric.py
  DistributeNodes.py
  ParMETISDistributeNodes.py
  MortonOrderDistributeNodes.py
  PeanoHilbertDistributeNodes.py
  VoronoiDistributeNodes.py
  NestedGridDistributeNodes.py
  GzipFileNodeGenerator.py
  DumpGzipFileNodeGenerator.py
  CubicNodeGenerator.py
  GenerateSphericalShellSection.py
  AbaqusNodeGenerator.py
  CentroidalVoronoiRelaxation.py
  InteriorGenerator.py
  FacetedSurfaceGenerator.py
  AsciiFileNodeGenerator.py
  HydroStaticProfile.py
  StretchedLattice2d.py
  StretchedLattice3d.py
  GenerateEqualMassSheets3d.py
  SiloPolyMeshGenerator.py
  GenerateRatioSlab.py
  GenerateRatioSphere.py
  GenerateNodeProfile.py
  CentroidalGenerator2d.py
  centroidalRelaxNodes.py
  MedialGenerator.py
  MultiScaleMedialGenerator.py
  RecursivePrimitiveRefinementGenerator.py
  SEAGenerator.py
  FacetedVolumeRejecters.py
  )

spheral_add_cxx_library(NodeGenerators)

list(APPEND SPHERAL_CXX_LIBS Spheral_NodeGenerators)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
