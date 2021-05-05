#find_package(Sphinx)
set(SPHINX_FOUND true)
set(SPHINX_EXEC "${python_DIR}/bin/sphinx-build" CACHE STRING "Path to sphinx documentation generator")
message("-- Sphinx ${SPHINX_EXEC}")
 
if(ENABLE_DOCS AND NOT SPHINX_EXEC STREQUAL "SPHINX_EXEC-NOTFOUND")
  if(NOT DEFINED SPHINX_THEME)
    set(SPHINX_THEME sphinx_rtd_theme)
  endif()

  if(NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
  endif()

  # configured documentation tools and intermediate build results
  set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

  # Sphinx cache with pickled ReST documents
  set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

  # HTML output directory
  set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)

  set(DOC_DEPENDS ${spheral_py_depends})
  if(pip_BUILD)
    set(DOC_DEPENDS ${DOC_DEPENDS} pip-modules)
  endif()

  add_custom_target(Spheral_docs ALL
    ${PYTHON_EXE} ${SPHINX_EXEC}
    -q -b html
    -c "${BINARY_BUILD_DIR}"
    -d "${SPHINX_CACHE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${SPHINX_HTML_DIR}"
    DEPENDS ${DOC_DEPENDS}
    COMMENT "Building HTML documentation with Sphinx")
endif()
