#!/bin/bash
FILE_PATH=$1
ARGS="${@:2}"
FILE_NAME=$(basename -- "$FILE_PATH")
VIDEO_NAME="${FILE_NAME%%.*}"
FRAMES_DIR="images/$VIDEO_NAME"
LABELS_DIR="labels/$VIDEO_NAME"
mkdir -p $FRAMES_DIR
mkdir -p $LABELS_DIR
rm -f $FRAMES_DIR/*
ffmpeg -i $FILE_PATH $ARGS $FRAMES_DIR/image-%06d.png -hide_banner

