#!/bin/bash

SUBJECT=$1

DATA_DIR="/media/rana/Balthazar/3Dreconstruction/manus/dataset"
EXP_DIR="outputs_classic/object"
ROOT_DIR="${DATA_DIR}/${SUBJECT}"
CALIB_DIR="${ROOT_DIR}/calib.object"

## For all the objects we want to train
  OBJS=("bags2" "bags3" "bags4" "books1" "books2" "books3" "bottle1" "bottle2" "bottle3" "bottle4" "color1" "color2" "color3" "color4" "color5" "cube" "fruits1" "fruits2" "fruit3" "fruits4" "jars1" "jars2" "jars3" "mug1" "mug2" "mug3" "pillow" "shoes1" "shoes2" "tech1" "tech2" "utensils1" "utensils2" "utensils3" "utensils4" )
  for OBJ_NAME in "${OBJS[@]}"
do
  echo "Training $OBJ_NAME"
  OBJ_DIR="${ROOT_DIR}/objects/${OBJ_NAME}"
  EXP_OBJ_DIR="${EXP_DIR}/${SUBJECT}/${OBJ_NAME}/"

  python main_classic.py --config-name OBJ_GAUSSIAN_classic \
      output_dir=$EXP_OBJ_DIR \
      trainer.loggers='[csv]' \
      trainer.mode='train' \
      trainer.project='object' \
      trainer.exp_name=$OBJ_NAME \
      trainer.gpus=[0] \
      trainer.num_workers=4 \
      train_dataset.opts.split_ratio=0.98 \
      trainer.pl_vars.accumulate_grad_batches=1 \
      trainer.pl_vars.max_steps=10000 \
      train_dataset.opts.subject=$SUBJECT \
      train_dataset.opts.root_dir=$OBJ_DIR \
      train_dataset.opts.params_dir=$CALIB_DIR \
      model.opts.opacity_reset_interval=500 \
      model.opts.densification_interval=100 \
      model.opts.percent_dense=0.00001 \
      model.opts.densify_grad_threshold=0.00003 \
      model.opts.remove_outliers_step=100000000 \
      model.opts.isotropic_scaling=false \
      model.opts.sample_size=30000 \
      model.opts.densify_until_step=20000

## Inference on novel views requires downloading the Blender
## Set the BLENDER_PATH in the inference-object.sh
bash scripts/train/inference_object_classic.sh $SUBJECT $OBJ_NAME $EXP_OBJ_DIR
done
