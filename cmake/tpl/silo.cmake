set(SILO_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(SILO_CACHE "${CACHE_DIR}/silo-4.10.2-bsd.tgz")
set(SILO_URL "https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2-bsd.tgz")
set(SILO_MD5 "MD5=60fef9ce373daf1e9cc8320cfa509bc5")
set(SILO_SRC_DIR ${SILO_PREFIX}/src/silo)

#set(${lib_name}_INCLUDES silo.h)
set(${lib_name}_libs libsiloh5.a)

if(${lib_name}_BUILD)

  if (EXISTS ${SILO_CACHE})
    set(SILO_URL ${SILO_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${SILO_PREFIX}
    URL ${SILO_URL} 
    URL_HASH ${SILO_MD5}
    DOWNLOAD_DIR ${CACHE_DIR}
    PATCH_COMMAND patch -t ${SILO_SRC_DIR}/config/config.guess ${PATCH_DIR}/config.guess-silo-4.10.2-bsd.patch &&
                  patch -t ${SILO_SRC_DIR}/config/config.sub   ${PATCH_DIR}/config.sub-silo-4.10.2-bsd.patch
    CONFIGURE_COMMAND env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ${SILO_SRC_DIR}/configure
                      --enable-shared=no
                      --enable-fortran=no
                      --with-hdf5=${HDF5_INSTALL_DIR}/include,${HDF5_INSTALL_DIR}/lib
                      --prefix=${${lib_name}_DIR}
                      --enable-silex=no
                      --enable-browser=yes
    BUILD_COMMAND make 
    INSTALL_COMMAND make install
    DEPENDS ${hdf5_build_dep}

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

endif()

