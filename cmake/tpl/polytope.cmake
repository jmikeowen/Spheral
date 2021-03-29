set(POLYTOPE_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(POLYTOPE_CACHE "${CACHE_DIR}/0.6.2.tar.gz")
set(POLYTOPE_URL "https://github.com/pbtoast/polytope/archive/0.6.2.tar.gz")
set(POLTOPE_SRC_DIR ${POLYTOPE_PREFIX}/src/boost)

set(${lib_name}_libs libpolytope.a)

set(POLYTOPE_DEPENDS boost)
set(POLYTOPE_USE_PYTHON On)

if(ENABLE_CXXONLY)
  set(POLYTOPE_USE_PYTHON Off)
else()
  list(APPEND POLYTOPE_DEPENDS python-install pip-modules ${spheral_py_depends})
endif()

if(${lib_name}_BUILD)

  if (EXISTS ${POLYTOPE_CACHE})
    set(POLYTOPE_URL ${POLYTOPE_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${POLYTOPE_PREFIX}
    PATCH_COMMAND patch -t ${POLYTOPE_PREFIX}/src/polytope/src/PYB11/CMakeLists.txt  ${PATCH_DIR}/polytope-PYB11-CMakeLists.patch

    URL ${POLYTOPE_URL}
    URL_HASH "MD5=${POLYTOPE_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_INSTALL_PREFIX=${${lib_name}_DIR} 
               -DPYBIND11_INCLUDE_DIRS=${PYBIND11_INSTALL_DIR}/include
               -DUSE_PYTHON=${POLYTOPE_USE_PYTHON}
               -DPYTHON_EXE=${PYTHON_EXE}
               -DBoost_INCLUDE_DIR=${BOOST_INSTALL_DIR}/include
               -DTESTING=Off
               DEPENDS ${POLYTOPE_DEPENDS}

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

  if(NOT ENABLE_CXXONLY AND NOT ENABLE_STATIC_CXXONLY)
    install(
      FILES ${${lib_name}_DIR}/lib/python2.7/site-packages/polytope/polytope.so
      DESTINATION ${PYTHON_SITE_PACKAGE_DIR}
      )
  endif()
endif()

