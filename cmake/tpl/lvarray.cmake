set(LVARRAY_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(LVARRAY_DIST "LvArray-v0.2.0.tar.gz")
set(LVARRAY_URL "https://github.com/GEOSX/LvArray/releases/download/v0.2.0/${LVARRAY_DIST}")
set(LVARRAY_CACHE "${CACHE_DIR}/${LVARRAY_DIST}")

set(${lib_name}_libs liblvarray.a)

if(${lib_name}_BUILD)

  if (EXISTS ${LVARRAY_CACHE})
    set(LVARRAY_URL ${LVARRAY_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${LVARRAY_PREFIX}
    URL ${LVARRAY_URL}
    URL_HASH "MD5=${LVARRAY_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CMAKE_ARGS ../raja/src/
               -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_FLAGS=-fPIC

               -DRAJA_DIR=${raja_DIR}
               -DCHAI_DIR=${chai_DIR}
               -DCAMP_DIR=${raja_DIR}

               -DENABLE_TESTS=OFF
               -DENABLE_EXAMPLES=OFF
               -DENABLE_BENCHMARKS=OFF
               -DENABLE_DOCS=OFF

               -DENABLE_CUDA=${ENABLE_CUDA}
               -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
               -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
               -DCMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
               -DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}

               -DCMAKE_INSTALL_PREFIX=${${lib_name}_DIR}

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
    DEPENDS raja chai
  )

endif()
