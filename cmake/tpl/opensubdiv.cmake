set(OPENSUBDIV_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(OPENSUBDIV_DIST "v3_3_0.zip")
set(OPENSUBDIV_URL "https://github.com/PixarAnimationStudios/OpenSubdiv/archive/${OPENSUBDIV_DIST}")
set(OPENSUBDIV_CACHE "${CACHE_DIR}/${OPENSUBDIV_DIST}")
set(OPENSUBDIV_SRC_DIR ${OPENSUBDIV_PREFIX}/src/opensubdiv/src)

set(${lib_name}_libs libosdCPU.a)

if(${lib_name}_BUILD)

  if (EXISTS ${OPENSUBDIV_CACHE})
    set(OPENSUBDIV_URL ${OPENSUBDIV_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${OPENSUBDIV_PREFIX}
    URL ${OPENSUBDIV_URL}
    URL_HASH "MD5=${OPENSUBDIV_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_FLAGS=-fPIC
               -DBUILD_SHARED_LIBS=OFF
               -DNO_PYTHON=ON
               -DNO_EXAMPLES=ON
               -DNO_REGRESSION=ON
               -DNO_OMP=ON
               -DNO_CUDA=ON
               -DNO_OPENGL=ON
               -DNO_CLEW=ON
               -DNO_TBB=ON
               -DICC_LOCATION=${ICC_LOCATION}
               -DCMAKE_INSTALL_PREFIX=${${lib_name}_DIR}
    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

endif()
