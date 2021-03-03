#!/usr/bin/env bash

#-----------------------------------------------------------------------------------
# ***********************
# Handle Named Arguments.
# ***********************

SRC_DIR=
INSTALL_DIR=
CMAKE_ARGS=

while getopts "s:i:h:D:C:" name;
do
  case $name in
  s)  SRC_DIR="$OPTARG"
      printf 'Option -s source_dir "%s" specified\n' "$SRC_DIR"
      if [ -d $PWD/$SRC_DIR ]
      then
        SRC_DIR=$PWD/$SRC_DIR
      fi
      ;;
  i)  INSTALL_DIR="$OPTARG"
      printf 'Option -i install_dir "%s" specified\n' "$INSTALL_DIR"
      if [ -d $PWD/$INSTALL_DIR ]
      then
        INSTALL_DIR=$PWD/$INSTALL_DIR
      else
        if [ ! -d $INSTALL_DIR ]
        then
          printf 'ERROR : Cannot find "%s": exiting.\n' "$INSTALL_DIR"
        fi
      fi
      ;;
  h)  HOST_CONFIGS_DIR="$OPTARG"
      printf 'Option -h host_configs_dir "%s" specified\n' "$HOST_CONFIGS_DIR"
      if [ -d $PWD/$HOST_CONFIGS_DIR ]
      then
        HOST_CONFIGS_DIR=$PWD/$HOST_CONFIGS_DIR
      fi
      ;;
  D)  CMAKE_ARGS=$CMAKE_ARGS"-D$OPTARG ";;
  C)  CMAKE_ARGS=$CMAKE_ARGS"-C $OPTARG ";;
  ?)  printf "Usage: %s: [-d Spheral source directory] [CMake args ...]\n" $0
        ;;
  esac
done

if [ ! -z "$SRC_DIR" ]; then
  printf 'Using source_dir "%s"\n' "$SRC_DIR"
else
  SRC_DIR=../..
fi
if [ ! -z "$INSTALL_DIR" ]; then
  printf 'Using install_dir "%s"\n' "$INSTALL_DIR"
else
  INSTALL_DIR=../install
fi
if [ ! -z "$HOST_CONFIGS_DIR" ]; then
  printf 'Using host_configs_dir "%s"\n' "$HOST_CONFIGS_DIR"
else
  HOST_CONFIGS_DIR=$SRC_DIR/host-configs
fi

if [ ! -z "$CMAKE_ARGS" ]; then
  printf "Additional CMake arguments are: %s\n" "$CMAKE_ARGS"
fi
shift $(($OPTIND - 1))

#-----------------------------------------------------------------------------------
