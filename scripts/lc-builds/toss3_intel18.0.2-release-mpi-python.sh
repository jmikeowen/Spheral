#!/usr/bin/env bash

SCRIPT_PATH=${0%/*}
. "$SCRIPT_PATH/utils/parse-args.sh"

# Inherit build directory name from script name
BUILD_SUFFIX="lc_$(TMP=${BASH_SOURCE##*/}; echo ${TMP%.*})"

rm -rf ${BUILD_SUFFIX} 2>/dev/null
mkdir -p ${BUILD_SUFFIX}/install
mkdir -p ${BUILD_SUFFIX}/build && cd ${BUILD_SUFFIX}/build

module load cmake/3.14.5
module load intel/18.0.2

cmake \
  ${SRC_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${HOST_CONFIGS_DIR}/lc-builds/toss3/intel18.0.2_tpl.cmake \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel/intel-18.0.2/bin/icpc \
  -DCMAKE_C_COMPILER=/usr/tce/packages/intel/intel-18.0.2/bin/icc \
  -DICC_LOCATION=/usr/tce/packages/intel/intel-18.0.2/ \
  -DENABLE_OPENMP=On \
  -DENABLE_MPI=On \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  $CMAKE_ARGS \
