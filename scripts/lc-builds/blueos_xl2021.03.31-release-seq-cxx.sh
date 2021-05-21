#!/usr/bin/env bash

SCRIPT_PATH=${0%/*}
. "$SCRIPT_PATH/utils/parse-args.sh"

# Inherit build directory name from script name
BUILD_SUFFIX="lc_$(TMP=${BASH_SOURCE##*/}; echo ${TMP%.*})"

rm -rf ${BUILD_SUFFIX} 2>/dev/null
mkdir -p ${BUILD_SUFFIX}/install
mkdir -p ${BUILD_SUFFIX}/build && cd ${BUILD_SUFFIX}/build

module load cmake/3.14.5
module load xl/2021.03.31

cmake \
  ../.. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/xl/xl-2021.03.31/bin/xlc \
  -DENABLE_OPENMP=Off \
  -DENABLE_MPI=Off \
  -DENABLE_CUDA=Off \
  -DENABLE_STATIC_CXXONLY=On \
  -DSPHERAL_TPL_DIR=$INSTALL_DIR \
  "$@" \
  #-DSPHERAL_TPL_DIR=/usr/workspace/wsrzd/davis291/SPHERAL/blueos_Spheral_gcc8_noMPI/install/tpl \
