set(CHAI_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(CHAI_DIST "chai-2.3.0.tar.gz")
set(CHAI_URL "https://github.com/LLNL/CHAI/releases/download/v2.3.0/${CHAI_DIST}")
set(CHAI_CACHE "${CACHE_DIR}/${CHAI_DIST}")

set(${lib_name}_libs libchai.a libumpire.a)

if(${lib_name}_BUILD)

  if (EXISTS ${CHAI_CACHE})
    set(CHAI_URL ${CHAI_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${CHAI_PREFIX}
    URL ${CHAI_URL}
    URL_HASH "MD5=${CHAI_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CMAKE_ARGS ../raja/src/
               -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_FLAGS=-fPIC

               -DENABLE_TESTS=OFF
               -DENABLE_EXAMPLES=OFF
               -DENABLE_BENCHMARKS=OFF
               -DENABLE_DOCS=OFF

               -DENABLE_CUDA=${ENABLE_CUDA}
               -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
               -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}

               -DCMAKE_INSTALL_PREFIX=${${lib_name}_DIR}

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

endif()
