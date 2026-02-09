#!/bin/bash

start_subject=$1
end_subject=$2
radius=$3
voxel_set=$4

for i in $(seq -f "%02g" $start_subject $end_subject); do
    i_pad=$(printf "%02.f" $i) #pad with leading zero
    echo "Running subject ${i_pad}"
    python 3_compute_searchlight_RSA.py $i_pad $radius $voxel_set
    echo "Finished subject ${i_pad}"
done