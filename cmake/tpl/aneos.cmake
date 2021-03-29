set(ANEOS_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(ANEOS_DIST "M-ANEOS-v1.0.tar.gz")
set(ANEOS_CACHE "${CACHE_DIR}/${ANEOS_DIST}")
set(ANEOS_URL "https://github.com/isale-code/M-ANEOS/releases/download/v1.0beta/${ANEOS_DIST}")
set(ANEOS_SRC_DIR "${ANEOS_PREFIX}/src/aneos/src")
set(ANEOS_DEST_DIR "${${lib_name}_DIR}/lib")

#set(${lib_name}_INCLUDES aneos.h)
set(${lib_name}_libs libaneos.a)

if (EXISTS ${ANEOS_CACHE})
  set(ANEOS_URL ${ANEOS_CACHE})
endif()

if (${lib_name}_BUILD)

  ExternalProject_add(${lib_name}
    PREFIX ${ANEOS_PREFIX}
    URL ${ANEOS_URL} 
    URL_HASH "MD5=${ANEOS_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    PATCH_COMMAND patch -t ${ANEOS_SRC_DIR}/Makefile ${PATCH_DIR}/MANEOS-v1.0_Makefile.patch
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make FORT=${CMAKE_Fortran_COMPILER} -C ${ANEOS_SRC_DIR}
    BUILD_IN_SOURCE 1
    INSTALL_DIR ${ANEOS_DEST_DIR}
    INSTALL_COMMAND cp -f ${ANEOS_SRC_DIR}/libaneos.a ${ANEOS_DEST_DIR}
    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
    )

endif()
