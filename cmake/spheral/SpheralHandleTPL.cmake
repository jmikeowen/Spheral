# Initialize commonly used paths during TPL installs
set(CACHE_DIR ${CMAKE_BINARY_DIR}/tpl/cache)
set(PATCH_DIR ${SPHERAL_ROOT_DIR}/src/tpl/patch)
set(TPL_CMAKE_DIR ${CMAKE_MODULE_PATH}/tpl)

# If a TPL takes a -j set this to launch a parallel build 
if (NOT TPL_PARALLEL_BUILD)
  set (TPL_PARALLEL_BUILD "1")
endif()

# Verboseness of TPL builds
if (TPL_VERBOSE)
  set(OUT_PROTOCOL_EP 0)
else()
  set(OUT_PROTOCOL_EP 1)
  set(OUT_PROTOCOL_PIP "-q")
endif()

# Set the build directory for TPL to default to BUILD/Spheral-tpl if SPHERAL_TPL_DIR
# is not set.
if (NOT SPHERAL_TPL_DIR)
  get_filename_component(DEFAULT_TPL_LOCATION ${CMAKE_BINARY_DIR}/Spheral/tpl ABSOLUTE)
else()
  if (NOT IS_ABSOLUTE ${SPHERAL_TPL_DIR})
    set(SPHERAL_TPL_DIR ${CMAKE_BINARY_DIR}/${SPHERAL_TPL_DIR})
  endif()
  get_filename_component(DEFAULT_TPL_LOCATION ${SPHERAL_TPL_DIR} ABSOLUTE)
endif()
message("Default TPL location : ${DEFAULT_TPL_LOCATION}\n")



#----------------------------------------------------------------------------------------
#                                   Spheral_Handle_TPL
#----------------------------------------------------------------------------------------

# -------------------------------------------
# VARIBALES THAT NEED TO BE PREVIOSLY DEFINED
# -------------------------------------------
# BUILD_TPL             : REQUIRED : Set at configure time defines weather to search or build tpl.
# <lib_name>_DIR        : OPTIONAL : The installation location of the tpl, if installed or not.
#                                    If not set and BUILD_TPL=On <lib_name> will be installed to a 
#                                    default loc. 

# ----------------------
# INPUT-OUTPUT VARIBALES
# ----------------------
# <lib_name>   : REQUIRED : The name of the target tpl
# <dep_list>   : REQUIRED : list that appends new target <lib_name> to itself.

# -----------------------
# OUTPUT VARIABLES TO USE - Made available implicitly after funciton call.
# -----------------------
# <lib_name>_libs : list of full paths to tpl lib files to be linked to.
#----------------------------------------------------------------------------------------

function(Spheral_Handle_TPL lib_name dep_list)

  string(TOUPPER ${lib_name} LIB_NAME)

  # Get the pure MD5 HASH string.
  set(HASH ${${LIB_NAME}_MD5})
                                                       
  # If we are not building the TPL ...
  if(NOT BUILD_TPL OR NOT ${lib_name}_BUILD)

    # If no location to search is sepcified, search default dir
    if (NOT ${lib_name}_DIR)
      set(${lib_name}_DIR ${DEFAULT_TPL_LOCATION}/${lib_name}/${HASH})
      message("${lib_name}_DIR not set.")
      message("Setting ${lib_name} search to default location : ${${lib_name}_DIR}")
    # else search the given dir
    else()
      message("${lib_name}_DIR set.")
      message("Searching ${lib_name} for : ${${lib_name}_DIR}")
    endif()

    # Set usually when BUILD_TPL is off so that everything defaults to off
    if(NOT BUILD_TPL)
      set(${lib_name}_BUILD Off)
    endif()

    set(${lib_name}_build_dep)

  # We are building the TPL ...
  else()

    message("Generating build process for ${lib_name}.")

    # If no location to search is sepcified, install in default dir
    if (NOT ${lib_name}_DIR)
      set(${lib_name}_DIR ${DEFAULT_TPL_LOCATION}/${lib_name}/${HASH})
      message("${lib_name}_DIR not set. Installing ${lib_name} to default location : ${${lib_name}_DIR}")
    # else install in the given dir
    else()
      message("${lib_name}_DIR set. Installing ${lib_name} to : ${${lib_name}_DIR}")
    endif()

    set(${lib_name}_build_dep ${lib_name})
  endif()

  # Default this flag for the TPL to be added as a BLT lib, 
  # in the <tpl>.cmake file we may change this so as not to
  # add a blt lib component
  set(${lib_name}_ADD_BLT_TARGET ON)

  # Include the actual <tpl>.cmake file
  include(${TPL_CMAKE_DIR}/${lib_name}.cmake)

  list(APPEND ${lib_name}_INCLUDES $<BUILD_INTERFACE:${${lib_name}_DIR}/include>)

  # Generate full path to lib file for output list
  set(${lib_name}_LIBRARIES )
  foreach(lib ${${lib_name}_libs})
    set(lib_abs "${${lib_name}_DIR}/lib/${lib}")
    list(APPEND ${lib_name}_LIBRARIES $<BUILD_INTERFACE:${lib_abs}>)

    # Check all necessary files exist during config time when not installing TPL
    if (NOT BUILD_TPL OR NOT ${lib_name}_BUILD)
      if (NOT EXISTS ${lib_abs})
        message(FATAL_ERROR "Cannot find ${lib} in ${${lib_name}_DIR} for TPL ${lib_name}.")
      else()
        message("Found: ${lib_abs}")
      endif()
    endif()
  endforeach()

  # Register any libs/includes under a blt dir for later use/depends
  blt_register_library(NAME blt_${lib_name}
                       INCLUDES ${${lib_name}_INCLUDES}
                       LIBRARIES ${${lib_name}_LIBRARIES}
                       TREAT_INCLUDES_AS_SYSTEM On
                       )

  # Add the blt target to a list of libs that can be depended on
  if (${lib_name}_ADD_BLT_TARGET)
    list(APPEND spheral_blt_depends blt_${lib_name})
    if (${lib_name}_BUILD)
      list(APPEND ${dep_list} ${lib_name})
    endif()
  endif()

  set(${lib_name}_DIR ${${lib_name}_DIR} PARENT_SCOPE)
  set(${dep_list} ${${dep_list}} PARENT_SCOPE)
  set(spheral_blt_depends ${spheral_blt_depends} PARENT_SCOPE)
  set(${lib_name}_build_dep ${${lib_name}_build_dep} PARENT_SCOPE)

  message("")

endfunction()
