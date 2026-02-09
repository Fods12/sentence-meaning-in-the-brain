#!/bin/bash

start_subject=$1
end_subject=$2
type=$3

for i in $(seq -f "%02g" $start_subject $end_subject); do
    i_pad=$(printf "%02.f" $i) #pad with leading zero
    output_file="logs/log_sub-${i_pad}-sims-${type}.txt"
    echo "Running subject ${i_pad}"
    python 2_compute_neuro_sims.py $i_pad $type &> $output_file 2>&1
    echo "Finished subject ${i_pad}"
done