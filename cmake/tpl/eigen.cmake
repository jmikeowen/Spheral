set(EIGEN_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(EIGEN_CACHE "${CACHE_DIR}/eigen-3.4.0-rc1.tar.gz")
set(EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0-rc1/eigen-3.4.0-rc1.tar.gz")

set(${lib_name}_libs )
if(${lib_name}_BUILD)

  if (EXISTS ${EIGEN_CACHE})
    set(EIGEN_URL ${EIGEN_CACHE})
  endif()

  file(MAKE_DIRECTORY ${${lib_name}_DIR}/include)

  ExternalProject_add(${lib_name}
    PREFIX ${EIGEN_PREFIX}/${EIGEN_TARGET}
    URL ${EIGEN_URL}
    URL_HASH "MD5=${EIGEN_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CONFIGURE_COMMAND sleep 1
    BUILD_COMMAND sleep 1
    INSTALL_COMMAND cp -r ${EIGEN_PREFIX}/src/eigen/Eigen ${${lib_name}_DIR}/include

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
  )

endif()
