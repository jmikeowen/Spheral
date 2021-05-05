.. include:: <isoamsa.txt>

###############################################
Obtaining, building, and installing Spheral
###############################################

Spheral minimally requires a C++11 compliant compiler.  If you use git to clone the Spheral source note Spheral includes a git submodule: `BLT <https://github.com/LLNL/blt>`_.  In order to ensure such submodules are properly downloaded when cloning Spheral be sure to use the ``--recurse-submodules`` or ``--recursive`` git options:

::

  git clone --recursive https://github.com/jmikeowen/Spheral

If you forget to use the ``--recursive`` argument or if you checkout from a different branch you must run:

::

  git submodule update --init --recursive

Basic Build
-----------

The default build will try to build Spheral with Python and MPI support. This will be enough to run Spheral from the locally built python and pass the test suite.

I like to keep my build & install files separate from the git cloned source, so I usually build in a completely separate directory as follows:

::

  git clone --recursive https://github.com/jmikeowen/Spheral
  mkdir -p Spheral_release/BUILD && cd Spheral_release/BUILD
  cmake -DCMAKE_INSTALL_PREFIX=`cd ..; pwd` ../../Spheral
  make -j<N> install
  ../spheral -c "import Spheral"

In this example we performed our build in the directory ``Spheral_release/BUILD``, and installed all binaries and libraries in ``Spheral_release``.  Note this includes the TPL libraries, which are downloaded under ``Spheral_release/BUILD`` and installed to ``Spheral_release``.  The final line is simply a test that the installed Python client can load the Spheral Python modules.

The somewhat obtuse command ``-DCMAKE_INSTALL_PREFIX=`chdir ..; pwd``` just specifies the install directory as the full path to ``Spheral_release``.  Alternatively you can specify this path explicitly, such as ``-DCMAKE_INSTALL_PREFIX=/usr/local/Spheral_release``, if that were the correct path.

.. note::
   Although Spheral is simply a set of Python modules, it installs in a Python virtual environment, so the script ``spheral`` installed at the top level of the install tree is designed to load the virtual environment on invocation, and then unload it on completion.

.. note::
   Many users of CMake like to place the build directory as a subdirectory of the cloned code, so many examples you'll see online use "``cmake ..``".  All that matters really is that the final path on the CMake command line point to the top of the source tree.

C++ Only Build
--------------

If you do not need to build the Python interface you can build the compiled C++ libraries alone with ``-DENABLE_CXXONLY=On``.  This will skip the Python wrapping stage of the build. 

By default Spheral builds the libraries as shared objects.  If instead you would like to build the C++ libraries as static libs use ``-DENABLE_STATIC_CXXONLY=On``.

Third party libraries and Spheral
---------------------------------

Upon first install third party libraries (TPL) tar files and source will be installed through an external network. TPLs are cached within the Spheral build directory tree for future builds off network. To completely turn off installation of TPL's use ``-DBUILD_TPLS=Off``.

For just the C++ compiled Spheral a number of TPLs are required (and automatically installed by default):

- Boost
- Python
- Eigen
- Pybind11
- Polytope
- HDF5
- Silo
- Qhull
- ANEOS
- Opensubdiv

There are also a number of Python packages that are automatically installed during the third party build, including:

- Scipy
- Sobol
- Cython
- Twine
- sphinx_rtd_theme
- h5py
- decorator
- Matplotlib
- mpi4py
- Numpy-stl

By default the Spheral TPLs are installed to the path specified on the CMake command line with ``-DCMAKE_INSTALL_PREFIX=<full path>``.

To pass a custom path to a TPL build use the ``-D<TPL-Name-Here>_DIR=<full path>`` option. CMake will assume the directory passed has the same structure as if it were built the library itself.  For example, to specify a preinstalled version of Boost:

::

  cmake -DBOOST_DIR=$HOME/my_boost_build_dir ...

To force a download and build from a specific TAR for a TPL use ``-D<TPL-Name-Here>_URL=<url path>``. This can be either a local file path or URL:

::

   cmake -DBOOST_URL=https://my.tarfiles.com/boost/boost-1.2.3.4.tar ...

OpenMP/MPI
----------

OpenMP and MPI support is handled through BLT.  Use the option flags ``-DENABLE_OPENMP`` and ``-DENABLE_MPI`` respectively, choosing ``ON`` or ``OFF`` as appropriate.  

CMake variables
--------------------

In this section we list the CMake variables that can be tweaked for a Spheral build.  Where appropriate the options are listed, with the default value in *italics*.

``CMAKE_BUILD_TYPE``   (Debug, *Release*, RelWithDebInfo, MinSizeRel)
  Choose the type of build -- for more information see the `CMake documentation <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_.

``CMAKE_INSTALL_PREFIX``
  The top-level path for installing Spheral include files, libraries, and any Python modules or documentation.  This is synonymous with and replaces the older ``SPHERAL_INSTALL_DIR``.

``ENABLE_CXXONLY`` (On, *Off*)
  Do not build python wrappers for Spheral.

``ENABLE_STATIC_CXXONLY`` (On, *Off*)
  Do not build python wrappers for Spheral. Build static library files for Spheral.

``BUILD_TPLS`` (*On*, Off)
  Option to install TPLs or not during configuration stage.

``<TPL-Name-Here>_DIR``
  Directory of previously built TPL.

``<TPL-Name-Here>_URL`` 
  URL or local path to zip/tar file of TPL to download and install.
  Defaults to CMake defined URL/cache,  see cmake/InstallLibraries.cmake

``<TPL-Name-Here>_BUILD`` (*On*, Off)
  Tell the build system to build or not to build the given TPL. A ``<TPL>_DIR`` must be provided, otherwise it will search a default directory.

``ENABLE_OPENMP`` (*On*, Off)
  Support for OpenMP.

``ENABLE_MPI`` (*On*, Off)
  Support for MPI.

``ENABLE_1D`` (*On*, Off)
  Build Spheral with 1D support.

``ENABLE_2D`` (*On*, Off)
  Build Spheral with 2D support.

``ENABLE_3D`` (*On*, Off)
  Build Spheral with 3D support.

``ENABLE_ANEOS`` (*On*, Off)
  Install the ANEOS (Analytics Equation of State) package along with the Spheral interface to it.  This is a legacy equation of state frequently used for geophysical materials.  See descriptions in the `iSALE <https://github.com/isale-code/M-ANEOS>`_ documentation.

``ENABLE_HELMHOLTZ`` (*On*, Off)
  Compile the included Helmholtz equation of state, typically used in astrophysical calculations. See a discussion `here <http://cococubed.asu.edu/code_pages/eos.shtml>`_.

``ENABLE_OPENSUBDIV`` (*On*, Off)
  Install the Opensubdiv library along with the Spheral interface to it.  Opensubdiv is a `Pixar provided library <https://github.com/PixarAnimationStudios/OpenSubdiv>`_, which Spheral uses to implement refinement of polyhedra for some specialized problem generation capabilities.

``ENABLE_TIMER`` (*On*, Off)
  Enable timer information from Spheral.

``BOOST_HEADER_ONLY`` (On, *Off*)
  Specify that the Boost third party library will be header only, no compiled libs.

``DBC_MODE`` (None, All, Pre)
  Set the compile time design by contract (DBC) mode for Spheral.  Design by contract statements are very useful developer tools, whereby the developer can insert tests in the code as they write it.  These statements are both useful for tracking down bugs with fine-grained testing throughout the code, as well as useful documentation in the code about what sort of conditions are expected to hold.

  +------+---------------------------------------------------------------------------------+
  | None | Design by contract not enforced                                                 |
  +------+---------------------------------------------------------------------------------+
  | All  | All design by contract (``REQUIRE``, ``ENSURE``, ``CHECK``) statements active   |
  +------+---------------------------------------------------------------------------------+
  | Pre  | Only prerequisites (``REQUIRE``) statements active                              |
  +------+---------------------------------------------------------------------------------+

  Note the default depends on the ``CMAKE_BUILD_TYPE``:

  ``CMAKE_BUILD_TYPE=Debug`` |xrArr| default ``DBC_MODE`` is ``All``

  In all other cases the default is ``None``.

  It is worth noting ``DBC_MODE=All`` is quite expensive at run time (of order 4x more), so this is not intended to be active for a release/production compilation of Spheral.

``ENABLE_WARNINGS`` (On, *Off*)
  Enable compiler warnings.

``ENABLE_BOUNDCHECKING`` (On, *Off*)
  If building with the Gnu compilers enable STL bound checking by passing -D_GLIBCXX_DEBUG=1 to the compiler. 
  Note, this is a very expensive option at runtime!

``ENABLE_DOCS`` (On, *Off*)
  Choose whether or not to build this documentation.

``SPHINX_EXECUTABLE``
  Specify where the Sphinx executable is that should be used to build documentation.  If not given, assumes the Spheral built Sphinx will be used.

``SPHINX_THEME`` (*sphinx_rtd_theme*)
  Give the Sphinx theme to use when generating documentation.  Default based on read the docs theme.

``SPHINX_THEME_DIR``
  Where to look for Sphinx themes.

Linux Ubuntu Notes
------------------

When building on any system a few basic utilities are assumed to be installed.  It's impossible to cover all the possible build environments, but a common case is a Linux Ubuntu install.  In our experience we need at least the following packages beyond the base system default, which can be easily accomplished using ``apt install``)::

    sudo apt install cmake g++ gfortran zlib1g-dev libssl-dev libbz2-dev libreadline-dev build-essential libncurses5-dev libgdbm-dev libnss3-dev libffi-dev wget tk tk-dev libsqlite3-dev texlive-latex-recommended texlive-latex-extra dvipng

Most of these requirements are for building a full-featured Python installation.  If you also want to build the MPI parallel enabled version of Spheral you need an MPI implementation such as OpenMPI or MPICH -- OpenMPI for instance can be installed by adding the Ubuntu package ``openmpi-bin`` to the above list.

Checking/updating CMake version
...............................

Unfortunately most recent versions of Ubuntu Linux (and derivatives such as Mint) come with an older version of CMake by default (typically something like CMake v3.10).  This is too out of date for a Spheral build, and therefore needs to be updated before configuring and building Spheral.  First, just to make sure you have this issue you should check the version of cmake that comes with your distribution::

    cmake --version

If the result is something less than version 3.18, it's worth updating before starting to configure Spheral.  How to accomplish this varies by platform, but for the common case of Ubuntu (and similar ``apt`` based distributions) something like the following should suffice.

1. First, remove any existing cmake installation using apt::

     sudo apt remove --purge cmake

2. Follow the directions on the `Kitware site at this link <https://apt.kitware.com/>`_ to add their repository for installing packages.

3. Install a current version of cmake with::

     sudo apt install cmake

Check the final version again to make sure you have what you expect::

     cmake --version

WSL2 Notes
-----------

The Windows Subsystem for Linux (WSL) is a useful method of development on Windows 10 based systems.  If going this route we recommend having at least WSL2 for best results -- the original version of WSL (WSL1) also functioned, but is `significantly` slower for jobs such as compilation.

For the most part using an Ubuntu based WSL environment works just using the Ubuntu notes above.  However, one aspect of WSL2 needs to be adjusted.  The build process requires a fair amount of memory (in particular for a few of the Python binding modules), so we recommend having at least 32GB of swap space available.  On WSL2 this is accomplished by creating a ``.wslconfig`` file in your Windows home directory containing at least the following lines::

    [wsl2]
    swap=32GB

Build Scripts & LC Notes
------------------------

Scripts for building on LC systems can be found in ``scripts/lc-builds/``. These scripts build some of the more common configurations on LC machines. They have the added benefit of utilizing pre installed TPLs on LC. The pre-installed TPL loactions are passed using the configuration CMake files found in ``host-config/``.  
By default the scripts are designed to be run from the spheral root directory, a full build and test looks as follows::

    cd <Spheral_Root_Dir>
    ./scripts/lc-builds/toss3_gcc-8.3.1-release-mpi-python.sh
    cd lc_toss3-gcc-8.3.1-rel-mpi-py/build
    make -j install
    ../install/atstest ../../tests/integration.ats

The build scripts support a couple of named arguments. 

 ===================== ===============================================
 Arguments             Brief description
 ===================== ===============================================
 -s                    Spheral source directory. Useful when building
                       from a directory other than Spheral's root 
                       directory.
 -i                    Installation directory. This overwrites the 
                       scripts default installation directory from
                       ``<script_name>/install`` to a user provided
                       directory.
 ===================== ===============================================

An example of a scirpt build and test using these arguments is shown below::

    cd <Other_Directory>
    <Script_Dir>/toss3_gcc-8.3.1-release-mpi-python.sh -s <Spheral_Root_Dir> -i <Install_Dir>
    cd lc_toss3-gcc-8.3.1-rel-mpi-py/build
    make -j install
    <Install_Dir>/atstest <Spheral_Root_Dir>/tests/integration.ats

When using the build scripts, additional CMake arguments can be passed. This can be useful for a variety of reasons; below are a few examples altering how the scripts find / build TPLs for Spheral with CMake arguments.

To *SEARCH* for an installed TPL somewhere else::

    ./scripts/lc-builds/toss3_gcc8.3.1-release-mpi-python.sh -Dboost_DIR=<Full_Path_To_Boost_Install>

To *BUILD* a local version of a TPL to the default installation location::

    ./scripts/lc-builds/toss3_gcc8.3.1-release-mpi-python.sh -Dboost_BUILD=On -Dboost_DIR=""

To *BUILD* a local version of a TPL to a custom installation location::

    ./scripts/lc-builds/toss3_gcc8.3.1-release-mpi-python.sh -Dboost_BUILD=On -Dboost_DIR=<Local_Dir_To_Install>
