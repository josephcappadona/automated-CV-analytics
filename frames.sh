#!/bin/bash
VIDEO_FILE_PATH=$1
FFMPEG_ARGS="${@:2}"

VIDEO_FILE_NAME=$(basename -- "$VIDEO_FILE_PATH")
VIDEO_NAME="${VIDEO_FILE_NAME%%.*}"

DATA_DIR="data"
FRAMES_DIR="$DATA_DIR/$VIDEO_NAME/images"
LABELS_DIR="$DATA_DIR/$VIDEO_NAME/labels"
mkdir -p $FRAMES_DIR
mkdir -p $LABELS_DIR

ffmpeg -i $VIDEO_FILE_PATH $FFMPEG_ARGS $FRAMES_DIR/image-%06d.png -hide_banner

