set(PYTHON_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(PYTHON_VERSION "2.7.18")
set(PYTHON_CACHE "${CACHE_DIR}/Python-${PYTHON_VERSION}.tgz")
set(PYTHON_SRC_DIR ${PYTHON_PREFIX}/src/python)
set(PYTHON_INSTALL_DIR ${${lib_name}_DIR})
set(PYTHON_EXE ${PYTHON_INSTALL_DIR}/bin/python)
set(PYTHON_URL "http://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz")
set(PYTHON_MD5 "MD5=38c84292658ed4456157195f1c9bcbe1")
set(PYTHON_SITE_PACKAGE_DIR ${PYTHON_INSTALL_DIR}/lib/python2.7/site-packages)

set(${lib_name}_libs libpython2.7.so)
set(${lib_name}_INCLUDES $<BUILD_INTERFACE:${PYTHON_INSTALL_DIR}/include/python2.7/>)

set(PYTHON_C_COMPILER ${CMAKE_C_COMPILER})
set(PYTHON_CXX_COMPILER ${CMAKE_CXX_COMPILER})

if(PYTHON_C_COMPILER MATCHES "mpicc$")
  execute_process(COMMAND ${CMAKE_C_COMPILER} -show
                  OUTPUT_VARIABLE MPICC_SHOW )
  string (REPLACE " " ";" MPICC_SHOW_LIST "${MPICC_SHOW}")
  list(GET MPICC_SHOW_LIST 0 PYTHON_C_COMPILER)
  message("${PYTHON_C_COMPILER}")
endif()
if(PYTHON_CXX_COMPILER MATCHES "mpicxx$")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -show
                  OUTPUT_VARIABLE MPICXX_SHOW )
  string (REPLACE " " ";" MPICXX_SHOW_LIST "${MPICXX_SHOW}")
  list(GET MPICXX_SHOW_LIST 0 PYTHON_CXX_COMPILER)
  message("${PYTHON_CXX_COMPILER}")
endif()

if(${lib_name}_BUILD)

  if (EXISTS ${PYTHON_CACHE})
    set(PYTHON_URL ${PYTHON_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${PYTHON_PREFIX}
    URL ${PYTHON_URL} 
    URL_HASH ${PYTHON_MD5}
    DOWNLOAD_DIR ${CACHE_DIR}
    CONFIGURE_COMMAND env CC=${PYTHON_C_COMPILER}
                          CXX=${PYTHON_CXX_COMPILER}
                          CFLAGS=-I${ZLIB_INSTALL_DIR}/include
                          LIBS=-lz
                      ${PYTHON_SRC_DIR}/configure
                          --with-cxx-main='${PYTHON_CXX_COMPILER}'
                          --disable-ipv6
                          --enable-shared
                          --prefix=${PYTHON_INSTALL_DIR}
                          LDFLAGS=-Wl,-L${ZLIB_INSTALL_DIR}/lib,-rpath=${PYTHON_INSTALL_DIR}/lib
    BUILD_COMMAND make 
    INSTALL_COMMAND make install
    DEPENDS ${zlib_build_dep}

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )
  set(${lib_name}-install-dep ${lib_name})

endif()

add_custom_target(
  ${lib_name}-install
  COMMAND ${PYTHON_EXE} -V &> python-version.log
  DEPENDS ${${lib_name}-install-dep}
)

list(APPEND spheral_py_depends 
  python-install)


set(${lib_name}_BUILD ${${lib_name}_BUILD} PARENT_SCOPE)
set(PYTHON_EXE ${PYTHON_EXE} PARENT_SCOPE)
set(PYTHON_INSTALL_DIR ${PYTHON_INSTALL_DIR} PARENT_SCOPE)
set(PYTHON_SITE_PACKAGE_DIR ${PYTHON_SITE_PACKAGE_DIR} PARENT_SCOPE)
