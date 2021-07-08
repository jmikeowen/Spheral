set(AXOM_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(AXOM_DIST "Axom-v0.3.3.tar.gz")
set(AXOM_URL "https://github.com/LLNL/axom/releases/download/v0.3.3/${AXOM_DIST}")
set(AXOM_CACHE "${CACHE_DIR}/${AXOM_DIST}")

set(${lib_name}_libs 
    libaxom.a 
   )

if(${lib_name}_BUILD)

  if (EXISTS ${AXOM_CACHE})
    set(AXOM_URL ${AXOM_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    PREFIX ${AXOM_PREFIX}
    URL ${AXOM_URL}
    URL_HASH "MD5=${AXOM_MD5}"
    DOWNLOAD_DIR ${CACHE_DIR}
    SOURCE_SUBDIR src/
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_FLAGS=-fPIC

               -DAXOM_ENABLE_TESTS=OFF
               -DAXOM_ENABLE_EXAMPLES=OFF
               -DAXOM_ENABLE_DOCS=OFF
               -DAXOM_ENABLE_INLET=Off
               -DAXOM_ENABLE_LUMBERJACK=Off
               -DAXOM_ENABLE_SLAM=Off
               -DAXOM_ENABLE_MINT=Off
               -DAXOM_ENABLE_PRIMAL=Off
               -DAXOM_ENABLE_SPIN=Off
               -DAXOM_ENABLE_QUEST=Off
               -DENABLE_TESTS=Off

               -DCONDUIT_DIR=${CONDUIT_INSTALL_DIR}
               -DHDF5_DIR=${hdf5_DIR}

               -DCMAKE_INSTALL_PREFIX=${${lib_name}_DIR}

    DEPENDS ${hdf5_build_dep} ${conduit_build_dep}

    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
    LOG_BUILD ${OUT_PROTOCOL_EP}
    LOG_INSTALL ${OUT_PROTOCOL_EP}
  )

endif()
