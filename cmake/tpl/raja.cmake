set(RAJA_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(RAJA_DIST "RAJA-v0.13.0.tar.gz")
set(RAJA_URL "https://github.com/LLNL/RAJA/releases/download/v0.13.0/${RAJA_DIST}")
set(RAJA_CACHE "${CACHE_DIR}/${RAJA_DIST}")

set(${lib_name}_libs libRAJA.a)

if(${lib_name}_BUILD)

  if (EXISTS ${RAJA_CACHE})
    set(RAJA_URL ${RAJA_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${RAJA_PREFIX}
    URL ${RAJA_URL}
    URL_HASH "MD5=${RAJA_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    CMAKE_ARGS ../raja/src/
               -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_FLAGS=-fPIC

               -DENABLE_TESTS=OFF
               -DENABLE_EXAMPLES=OFF
               -DENABLE_EXERCISES=OFF
               -DENABLE_DOCS=OFF
               -DENABLE_QUEST=Off
               -DENABLE_TESTS=Off

               -DENABLE_CUDA=${ENABLE_CUDA}
               -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
               -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
               -DCUDA_ARCH=${CUDA_ARCH}
               -DCMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}

               -DCMAKE_INSTALL_PREFIX=${${lib_name}_DIR}

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

endif()
