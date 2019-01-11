#!/bin/bash

if (( $# < 1 )); then
    echo "USAGE:  frames.sh VIDEO_FILE_PATH [FFMPEG_ARGS ...]"
    exit 1
fi

VIDEO_FILE_PATH=$1
FFMPEG_ARGS="${@:2}"

VIDEO_FILE_NAME=$(basename -- "$VIDEO_FILE_PATH")
VIDEO_NAME="${VIDEO_FILE_NAME%%.*}"

DATA_DIR="data"
FRAMES_DIR="$DATA_DIR/$VIDEO_NAME"
mkdir -p $FRAMES_DIR

FFMPEG_COMMAND="ffmpeg -i $VIDEO_FILE_PATH $FFMPEG_ARGS $FRAMES_DIR/image-%06d.png -hide_banner"
echo "$FFMPEG_COMMAND" > "$DATA_DIR/$VIDEO_NAME.command"
eval $FFMPEG_COMMAND

