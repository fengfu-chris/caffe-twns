#!/bin/bash

CAFFE_ROOT=../../
GPU_ID=0
PRECISION=ternary # other choices: binary, single (default)
DELTA=7  # will divide 10 in the ternary function 
DEBUG=no
PREFFIX='lenet_tn'
SOLVER=${PREFFIX}_solver.prototxt 
LOG=logs/train_${PREFFIX}_`date +%Y-%m-%d-%H-%M`.log

$CAFFE_ROOT/build/tools/caffe train \
--gpu=${GPU_ID} \
--precision=${PRECISION} \
--delta=${DELTA} \
--solver=${SOLVER} \
--debug=${DEBUG} \
2>&1 | tee $LOG
