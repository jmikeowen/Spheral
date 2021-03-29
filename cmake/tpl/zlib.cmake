set(ZLIB_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(ZLIB_DIST "zlib-1.2.11.tar.gz")
set(ZLIB_CACHE "${CACHE_DIR}/${ZLIB_DIST}")
set(ZLIB_URL "http://zlib.net/${ZLIB_DIST}")
set(ZLIB_SRC_DIR ${ZLIB_PREFIX}/src/zlib)

set(${lib_name}_libs libz.a)

if(${lib_name}_BUILD)

  if (EXISTS ${ZLIB_CACHE})
    set(ZLIB_URL ${ZLIB_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${ZLIB_PREFIX}
    URL ${ZLIB_URL} 
    URL_HASH "MD5=${ZLIB_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CONFIGURE_COMMAND env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=-fpic ${ZLIB_SRC_DIR}/configure
                      --prefix=${${lib_name}_DIR}
    BUILD_COMMAND make 
    INSTALL_COMMAND make install

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

endif()

set(ZLIB_INSTALL_DIR ${${lib_name}_DIR} PARENT_SCOPE)
