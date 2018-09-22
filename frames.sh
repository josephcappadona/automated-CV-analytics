#!/bin/bash
mkdir images
FILE_PATH=$1
FILE_NAME=$(basename -- "$FILE_PATH")
VIDEO_NAME="${FILE_NAME%%.*}"
DIR="images/$VIDEO_NAME"
mkdir $DIR
ffmpeg -i $1 -ss 00:00:00 -vf fps=5 $DIR/image-%06d.png

