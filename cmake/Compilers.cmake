if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS -wd11074,11076,654)
  set(SPHERAL_PYB11_TARGET_FLAGS )
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS -Wno-undefined-var-template)
  set(SPHERAL_PYB11_TARGET_FLAGS
    "-Wno-unused-local-typedefs"
    "-Wno-self-assign-overloaded"
    "-Wno-overloaded-virtual"
    "-Wno-delete-non-abstract-non-virtual-dtor")
endif()
