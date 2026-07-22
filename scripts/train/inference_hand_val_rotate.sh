#!/bin/bash

SUBJECT=$1
EXP=$2
ACTION=$3
FRAME=$4
MAX_VIEWS=$5

if [ -z "$SUBJECT" ] || [ -z "$EXP" ]; then
  echo "Usage: bash scripts/train/inference_hand_val_rotate.sh <subject> <exp> [action] [frame] [max_views]"
  exit 1
fi

CMD="python scripts/process/render_pose_multiview_rotate.py --subject $SUBJECT --exp $EXP"
if [ -n "$ACTION" ] && [ -n "$FRAME" ]; then
  CMD="$CMD --action $ACTION --frame $FRAME"
fi
if [ -n "$MAX_VIEWS" ]; then
  CMD="$CMD --max_views $MAX_VIEWS"
fi

conda run -n manus $CMD
