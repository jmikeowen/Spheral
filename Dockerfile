###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

FROM spheral/tpl:gcc-8 AS gcc8
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace

RUN mkdir -p spheral_release/build spheral_release/install
RUN cd spheral_release/build && cmake ../../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran-8 -DCMAKE_INSTALL_PREFIX="../install" -DENABLE_MPI=Off -DBUILD_TPL=Off -DSPHERAL_TPL_DIR=/home/axom/workspace/spheral_tpl/install/tpl
RUN cd spheral_release/build && make -j4 SpheralCXXTypes
RUN cd spheral_release/build && make install
RUN ./spheral_release/install/spheral -c "import Spheral"
