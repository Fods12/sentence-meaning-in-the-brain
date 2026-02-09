model="parse-GPT4_CN_struct_control-2_set-all"
3dmerge -gmean -prefix "second_level_results/${model}_mean.nii.gz" */*"${model}.nii.gz"
3dcalc -a second_level_results/${model}_mean.nii.gz -b masks/Glasser-cortex.nii.gz -expr 'a*step(b)' \
       -prefix "second_level_results/${model}_masked_mean.nii.gz"