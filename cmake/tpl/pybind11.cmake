set(PYBIND11_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(PYBIND11_CACHE "${CACHE_DIR}/v2.4.3.tar.gz")
set(PYBIND11_URL "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz")

set(${lib_name}_libs )
if(${lib_name}_BUILD)

  if (EXISTS ${PYBIND11_CACHE})
    set(PYBIND11_URL ${PYBIND11_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${PYBIND11_PREFIX}/${lib_name}
    URL ${PYBIND11_URL}
    URL_HASH "MD5=${PYBIND11_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CMAKE_ARGS -DPYBIND11_TEST=Off
               -DCMAKE_INSTALL_PREFIX=${${lib_name}_DIR}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

endif()
set(PYBIND11_INSTALL_DIR ${${lib_name}_DIR} PARENT_SCOPE)
