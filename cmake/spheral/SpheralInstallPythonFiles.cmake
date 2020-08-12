#-----------------------------------------------------------------------------------
# spheral_install_python_files
#     - Copies Python files to the Spheral install path, and byte compiles them
#
# The list of python files should be passed as the arguments
#
# Note, if ENABLE_CXXONLY is set, this function does nothing
#-----------------------------------------------------------------------------------

function(spheral_install_python_files)

  if (NOT ENABLE_CXXONLY)
    install(FILES ${ARGV}
            DESTINATION Spheral)
    install(CODE "execute_process( \
            COMMAND ${SPHERAL_INSTALL_DIR}/python/bin/python -m compileall Spheral \
            WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX})")
  endif()

endfunction()
