#!/usr/bin/env bash

SCRIPT_PATH=${0%/*}
. "$SCRIPT_PATH/utils/parse-args.sh"

# Inherit build directory name from script name
BUILD_SUFFIX="lc_$(TMP=${BASH_SOURCE##*/}; echo ${TMP%.*})"

rm -rf ${BUILD_SUFFIX} 2>/dev/null
mkdir -p ${BUILD_SUFFIX}/install
mkdir -p ${BUILD_SUFFIX}/build && cd ${BUILD_SUFFIX}/build

module load cmake/3.14.5
module load gcc/8.3.1

cmake \
  ${SRC_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/gcc/gcc-8.3.1/bin/g++ \
  -C ${SRC_DIR}/host-configs/lc-builds/toss3/gccX_tpl.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_MPI=Off \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  $CMAKE_ARGS \
