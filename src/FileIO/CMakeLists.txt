include_directories(.)
set(FileIO_sources
    FileIO.cc
    FlatFileIO.cc
    SiloFileIO.cc
    PyFileIO.cc
    vectorstringUtilities.cc
    )

set(FileIO_headers
    DbFileIO.hh
    FileIO.hh
    FileIOInline.hh
    HDF5IO.hh
    HDF5Traits.hh
    HDF5Types.hh
    )

spheral_install_python_files(
  GzipFileIO.py
  PlyFileIO.py
  )

spheral_add_cxx_library(FileIO)

list(APPEND SPHERAL_CXX_LIBS Spheral_FileIO)
set(SPHERAL_CXX_LIBS "${SPHERAL_CXX_LIBS}" PARENT_SCOPE)
