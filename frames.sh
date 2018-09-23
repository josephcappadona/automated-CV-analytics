#!/bin/bash
FILE_PATH=$1
ARGS="${@:2}"
FILE_NAME=$(basename -- "$FILE_PATH")
VIDEO_NAME="${FILE_NAME%%.*}"
DIR="images/$VIDEO_NAME"
mkdir -p $DIR
rm -f $DIR/*
ffmpeg -i $FILE_PATH $ARGS $DIR/image-%06d.png -hide_banner

