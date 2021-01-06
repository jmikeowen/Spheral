macro(Install_Python_distutils_library lib_name lib_file lib_url lib_dir)
  add_custom_command(
    OUTPUT ${lib_file}
    COMMAND echo "-- downloading ${lib_file}"
    COMMAND wget ${lib_url} --no-check-certificate -O ${lib_file} > ${lib_file}-download.log
    COMMAND tar -zxvf ${lib_file} > ${lib_name}-unpack.log
    )
  add_custom_target(${lib_name}
    COMMAND ${PYTHON_EXE} setup.py install > ${lib_name}-install.log
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${lib_dir}
    COMMENT "Installing ${lib_name}"
    DEPENDS python-install pip-modules ${lib_file}
    )
  list(APPEND spheral_py_depends ${lib_name})
endmacro()
