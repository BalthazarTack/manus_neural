#!/bin/bash

SUBJECT_NAME=$1
EXP_NAME=$2
DATA_DIR="/media/rana/Balthazar/3Dreconstruction/manus/dataset"
EXP_DIR="/media/rana/Balthazar/3Dreconstruction/manus/outputs/hand/${SUBJECT_NAME}/${EXP_NAME}"
echo "EXP DIR: $EXP_DIR"
echo "Subject name: $SUBJECT_NAME"

OUT_PATH="${EXP_DIR}/results/novel_cam.pkl"
PC_PLY_PATH="${EXP_DIR}/init_gaussians.ply"

if test -f $OUT_PATH; then
   echo "Novel cam file exists!!!"
else
  # Get novel cam path in the Blender
/opt/blender-4.1/blender \
   ./data/blend_files/static.blend \
   -P scripts/process/bl_render.py -b -- $PC_PLY_PATH $OUT_PATH 0.8 0.005 0 0 0
fi

## Inference on the test/train dataset
python main.py --config-name config.yaml \
    --config-path $EXP_DIR \
    hydra.run.dir=$EXP_DIR \
    trainer.mode='test' \
    trainer.project='hand' \
    'checkpoint="best"' \
    test_dataset.opts.resize_factor=1.0 \
    test_dataset.opts.frame_sample_rate=1 \
    test_dataset.opts.cam_path=$OUT_PATH \
    test_dataset.opts.color_bkgd_aug="white" \
    test_dataset.opts.test_on_canonical_pose=true \
    test_dataset.opts.worst_cases=false \
    test_dataset.opts.metadata_path="/media/rana/Balthazar/3Dreconstruction/manus/preprocess/pose_outputs/subject1/grasps/color1_grasp1/meta_data.pkl" \
