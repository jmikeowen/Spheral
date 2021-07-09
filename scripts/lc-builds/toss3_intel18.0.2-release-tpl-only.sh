#!/usr/bin/env bash

SCRIPT_PATH=${0%/*}
. "$SCRIPT_PATH/utils/parse-args.sh"

# Inherit build directory name from script name
BUILD_SUFFIX="lc_$(TMP=${BASH_SOURCE##*/}; echo ${TMP%.*})"

rm -rf ${BUILD_SUFFIX}/build 2>/dev/null
mkdir -p ${BUILD_SUFFIX}/install
mkdir -p ${BUILD_SUFFIX}/build && cd ${BUILD_SUFFIX}/build

module load cmake/3.14.5
module load intel/18.0.2

cmake \
  ${SRC_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel/intel-18.0.2/bin/icpc \
  -DCMAKE_C_COMPILER=/usr/tce/packages/intel/intel-18.0.2/bin/icc \
  -DICC_LOCATION=/usr/tce/packages/intel/intel-18.0.2/ \
  -DENABLE_OPENMP=On \
  -DENABLE_MPI=On \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DBUILD_TPL_ONLY=On \
  $CMAKE_ARGS \

cd $BUILD_SUFFIX/build
make -j 16
make install

cd -
find ${BUILD_SUFFIX}/ -type d -exec chmod g+rx {} \;
find ${BUILD_SUFFIX}/ -type f -exec chmod g+rx {} \;
find ${BUILD_SUFFIX}/ -name "*egg-info" -exec chgrp wciuser {} \;
