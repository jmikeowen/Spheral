set(MHD_inst
  )

set(MHD_sources
    MHDFieldNames.cc
    #ConductingFluidNodeList.cc
    CurrentDensityUpdatePolicy.cc
    SpitzerResistivityUpdatePolicy.cc
    MagnetosonicSpeedUpdatePolicy.cc
    #PriceMonaghanDissipation.cc
    MHD.cc 
    )
instantiate(MHD_inst MHD_sources)

set(MHD_headers
    #ConductingFluidNodeList.hh
    CurrentDensityUpdatePolicy.hh
    MagnetosonicSpeedUpdatePolicy.hh
    MHD.hh
    MHDFieldNames.hh
    #PriceMonaghanDissipation.hh
    SpitzerResistivityUpdatePolicy.hh
    )

set(MHD_includes
    ${SPHERAL_SOURCE_DIR}
    ${SPHERAL_SOURCE_DIR}/MHD
    ${EIGEN_LIB_LOCATION}
    )


blt_add_library(NAME        Spheral_MHD
                HEADERS     ${MHD_headers}
                SOURCES     ${MHD_sources}
                DEPENDS_ON  ${spheral_depends}
                INCLUDES    ${MHD_includes}
                )

install(TARGETS             Spheral_MHD
        EXPORT              ${PROJECT_NAME}-targets
        DESTINATION         lib
        INCLUDES DESTINATION include
        )

install(FILES       ${MHD_headers}
        DESTINATION include/MHD
        )

