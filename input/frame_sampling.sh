#!/bin/bash

dir="$PWD/videos/*.avi"
for fullfilename in $dir; do
	filename=$(basename "$fullfilename")
	fname="${filename%.*}"
	result_fname=${PWD}/train/${fname}_%04d.jpg
    ffmpeg -i $fullfilename -vf "select=not(mod(n\,10))" -vsync vfr -q:v 2 $result_fname
done
