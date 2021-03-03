include_directories(.)
set(NodeList_inst
    ASPHSmoothingScale
    FluidNodeList
    NodeList
    NodeListRegistrar
    SPHSmoothingScale
    SmoothingScaleBase
    SolidNodeList
    generateVoidNodes
    FixedSmoothingScale
    nthNodalMoment
   )

set(NodeList_sources )

instantiate(NodeList_inst NodeList_sources)

set(NodeList_headers
    ASPHSmoothingScale.hh
    FluidNodeList.hh
    FluidNodeListInline.hh
    FluidNodeTraits.hh
    NodeList.hh
    NodeListInline.hh
    NodeListRegistrar.hh
    NodeListRegistrarInline.hh
    SPHSmoothingScale.hh
    SmoothingScaleBase.hh
    SolidNodeList.hh
    SolidNodeListInline.hh
    generateVoidNodes.hh
    secondMomentUtilities.hh
    )

spheral_install_python_files(
  FluidNodeLists.py
  SolidNodeLists.py
  VoidNodeLists.py
  )

spheral_add_cxx_library(NodeList)

