# General rule for installing pip modules. Downloads necessary tar/wheel files to the cache dir
# then installs in the future from these files so that offline builds are possible.
#
# NOTE:
# Since we are still on Python 2.7.*, we have to make a few compromises here:
#   1. We lock the exact versions of PIP packages we're using to the older known Python 2 friendly versions.
#   2. We removed the --upgrade option from the PIP install command.
# Hopefully once we move to Python 3 we can relax this stringent version locking and just stay up to date.

macro(Install_PipModules lib_list)
  set(${lib_list}_stamp_file "${CACHE_DIR}/.${lib_list}_pip_download.stamp")
  add_custom_command(
    OUTPUT ${${lib_list}_stamp_file}
    COMMAND echo "-- pip downloading ${lib_list}"
    COMMAND ${PYTHON_EXE} ${PIP_EXE} ${OUT_PROTOCOL_PIP} download --no-binary :all -d ${CACHE_DIR} ${${lib_list}}
    COMMAND touch ${${lib_list}_stamp_file}
  )
  add_custom_target(
    ${lib_list}
    COMMAND echo "-- pip installing ${lib_list}"
    COMMAND ${PYTHON_EXE} ${PIP_EXE} ${OUT_PROTOCOL_PIP} install ${${lib_list}} --no-index --find-links ${CACHE_DIR}
    DEPENDS pip-install ${${lib_list}_DEPENDS} ${${lib_list}_stamp_file}
  )
endmacro()

# Pip setup modules these need to be 
# installed before everything else
set(pip-setup-modules
    setuptools
    wheel)

# General pip modules, anything from PyPi can 
# be added to this list to install
set(pip-modules
    numpy-stl==2.11.2
    PYB11Generator
    matplotlib==2.2.5
    decorator==4.4.2
    h5py==2.10.0
    sphinx==1.8.5
    sphinx_rtd_theme==0.5.0
    twine==1.15.0
    cython==0.29.21
    sobol==0.9
    scipy==1.2.3
    pipreqs==0.4.10
    virtualenv==20.2.2)

# Only needed when we're allowing MPI parallelism
if (ENABLE_MPI)
  list(APPEND pip-modules mpi4py)
endif()

# Custom python modules to install that need their
# own install process these will be installed last 
# as their own dedicated targets
set(pip-custom-modules
    gnuplot
    ats)

# pip modules need to depend on the pip-setup-modules 
# being installed before the others
set(pip-modules_DEPENDS pip-setup-modules)
set(pip-modules_stamp_file_DEPENDS pip-setup-modules)

if(pip_BUILD)
  # Run install macro for PyPi modules
  Install_PipModules(pip-setup-modules)
  Install_PipModules(pip-modules)

  # Run install for custom modules with thier own .cmake file
  foreach(lib_name ${pip-custom-modules})
    include(${TPL_CMAKE_DIR}/${lib_name}.cmake)
  endforeach()
endif()

