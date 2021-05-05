set(HDF5_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(HDF5_CACHE "${CACHE_DIR}/hdf5-1.10.4.tar.bz2")
set(HDF5_URL "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.4/src/hdf5-1.10.4.tar.bz2")

set(${lib_name}_libs libhdf5.a libhdf5_hl.a)

if(${lib_name}_BUILD)

  if (EXISTS ${HDF5_CACHE})
    set(HDF5_URL ${HDF5_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${HDF5_PREFIX}
    URL ${HDF5_URL} 
    URL_HASH "MD5=${HDF5_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CONFIGURE_COMMAND env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=-fPIC CXXFLAGS=-fPIC FCFLAGS=-fPIC ${HDF5_PREFIX}/src/hdf5/configure --prefix=${${lib_name}_DIR}
    BUILD_COMMAND make 
    INSTALL_COMMAND make install

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

endif()
set(HDF5_INSTALL_DIR ${${lib_name}_DIR} PARENT_SCOPE)
