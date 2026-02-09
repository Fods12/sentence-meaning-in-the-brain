#!/bin/bash

start_subject=$1
end_subject=$2
fulL_run=$3

for ((i=$start_subject; i<=$end_subject; i++)); do
    i_pad=$(printf "%02.f" $i) #pad with leading zero
    echo "Running subject ${i_pad}"
    output_file="logs/log_sub-${i_pad}-nii2.txt"
    python 1_compute_GLMsingle_Fodor2024_nii.py $i_pad $fulL_run &> $output_file 2>&1
    echo "Finished subject ${i_pad}"
done