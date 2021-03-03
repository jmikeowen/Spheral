set(FVCRKH_inst
  FVCRKHydroBase
  )

set(FVCRKH_sources )
instantiate(FVCRKH_inst FVCRKH_sources)

set(FVCRKH_headers
  FVCRKHydroBase.hh
  )

set(FVCRKH_includes
  ${SPHERAL_SOURCE_DIR}
  ${SPHERAL_SOURCE_DIR}/FVCRKH
  ${EIGEN_LIB_LOCATION}
  )


blt_add_library(NAME        Spheral_FVCRKH
                HEADERS     ${FVCRKH_headers}
                SOURCES     ${FVCRKH_sources}
                DEPENDS_ON  ${spheral_depends}
                INCLUDES    ${FVCRKH_includes}
                )

install(TARGETS             Spheral_FVCRKH
        EXPORT              ${PROJECT_NAME}-targets
        DESTINATION         lib
        INCLUDES DESTINATION include
        )

install(FILES       ${FVCRKH_headers}
        DESTINATION include/FVCRKH
        )

