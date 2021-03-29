set(PIP_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${lib_name})
set(PIP_DIST pip-9.0.1-py2.py3-none-any.whl)
set(PIP_URL "https://pypi.python.org/packages/b6/ac/7015eb97dc749283ffdec1c3a88ddb8ae03b8fad0f0e611408f196358da3/pip-9.0.1-py2.py3-none-any.whl")
set(PIP_CACHE ${CACHE_DIR}/${PIP_DIST})
set(PIP_EXE ${PYTHON_INSTALL_DIR}/bin/pip2.7)

if(${lib_name}_BUILD)

  if (EXISTS ${PIP_CACHE})
    set(PIP_URL ${PIP_CACHE})
  endif()

  ExternalProject_add(${lib_name}
    URL ${PIP_URL}
    URL_HASH "MD5=${PIP_MD5}"
    DOWNLOAD_NO_EXTRACT TRUE
    DOWNLOAD_DIR ${CACHE_DIR}
    PREFIX ${PIP_PREFIX}
    CONFIGURE_COMMAND ${PYTHON_EXE} ${CACHE_DIR}/${PIP_DIST}/pip install --no-index ${CACHE_DIR}/${PIP_DIST}
    BUILD_COMMAND sleep 1
    INSTALL_COMMAND sleep 1
    DEPENDS python-install
    LOG_DOWNLOAD ${OUT_PROTOCOL_EP}
    LOG_CONFIGURE ${OUT_PROTOCOL_EP}
  )
  set(${lib_name}-install-dep ${lib_name})

else()
  set(${lib_name}_ADD_BLT_TARGET OFF)
endif()

add_custom_target(
  ${lib_name}-install
  COMMAND ${PYTHON_EXE} ${PIP_EXE} -V &> pip-version.log
  DEPENDS ${${lib_name}-install-dep}
)

list(APPEND ${dep_list}
  pip-install)

set(PIP_EXE ${PIP_EXE} PARENT_SCOPE)
