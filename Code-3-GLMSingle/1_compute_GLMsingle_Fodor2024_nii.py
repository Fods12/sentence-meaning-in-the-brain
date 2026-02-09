## Code to run GLMsingle on the fMRI data and save outputs as .nii files for each subject
# Script adapted from https://github.com/cvnlab/GLMsingle/tree/main/examples
# James Fodor 2024

# Load and import packages
import numpy as np
import scipy.io as sio
import nibabel
import os
import time
import warnings
import json
import sys
import itertools
import nilearn
import shutil

from os.path import join, exists, split
from tqdm import tqdm
from pprint import pprint
from glmsingle.glmsingle import GLM_single
from nilearn.masking import apply_mask

# Remove annoying warnings
warnings.filterwarnings('ignore')

# Get command line parameters
sub_id = str(sys.argv[1])  # subject ID
run_full = int(sys.argv[2]) # 1=run full analysis, 0=just load existing data
print('subject:',sub_id)



### ------------------------------------------Prepare data for processing------------------------------------------

## Create directories

# get path to the directory to which GLMsingle was installed
homedir = os.getcwd()

# create directory for saving data
output_dir = join(homedir,'sub-'+sub_id+'-nii')
if run_full:
    print('full analysis')
    os.makedirs(output_dir,exist_ok=True)
else:
    print('partial analysis')

print('Output dir:',output_dir)



## Load preprocessed BOLD data

# set input directory
input_dir = "../Data-fmri/preprocessed/sub-"+sub_id+"/func/"
print('Input dir:',input_dir)

# get filenames
datafiles = [file for file in os.listdir(input_dir) if ('bold.nii' in file)]
datafiles.sort()
print(datafiles)

# load BOLD data from .nii files
if run_full:
    data = []
    for file in tqdm(range(len(datafiles))):
        tmp = nibabel.load(os.path.join(input_dir,datafiles[file]))
        data.append(tmp.get_fdata())
    print(data[0].shape)

# load stimuli indices file
with open(input_dir+"/sub-"+sub_id+"_sentence_end_indices.json", "r") as file:
    sentence_end_indices = json.load(file)
with open(input_dir+"/sub-"+sub_id+"_sentence_start_indices.json", "r") as file:
    sentence_start_indices = json.load(file)


## Specify key scan parameters
num_stimuli = len(list(sentence_end_indices.values()))
num_runs = len(datafiles)
num_tr = 636
stim_duration = 3 # in seconds
tr = 0.8 # TR in seconds
print('Num stimuli:',num_stimuli)
print('Stim duration:',stim_duration)



## Construct design matrix from stored json files

# loop over runs
design = [[] for i in np.arange(num_runs)]
for run_id in np.arange(num_runs):
    design_this_run = []
    run_start_indices = np.array(list(sentence_start_indices.values()))[:,run_id] # get start indices for this run
    run_end_indices = np.array(list(sentence_end_indices.values()))[:,run_id] # get end indices for this run

    # loop over stimuli in run
    for stim_id in np.arange(0,num_stimuli):
        stim_time_series = np.zeros(num_tr) # initialise a time series of all zeros
        stim_start_index = run_start_indices[stim_id] # get start index for each sentence
        stim_end_index = run_end_indices[stim_id] # get end index for each sentence
        sentence_duration = stim_end_index-stim_start_index # number of TRs each sentence was shown for
        if stim_end_index>0: # only if this stimulus appears in this run
            if sentence_duration<=4:
                stim_time_series[stim_start_index] = 1 # start at stimulus onset
            else:
                time_adj = int((sentence_duration-stim_duration)/2)
                stim_time_series[stim_start_index+time_adj] = 1 # push start time a bit to the middle for longer sents
        design_this_run.append(stim_time_series)
    
    # convert to numpy array
    design_this_run = np.array(design_this_run)
    design[run_id] = design_this_run.T # need to transpose to get in right format

print('Num runs:',len(design))
print('Design matrix:',design[0].shape)



## Extract indices for each condition

# Construct a vector containing condition ids in chronological order
design_concat = np.concatenate(design,axis=0)
sent_id_ordered = []

for TR in range(design_concat.shape[0]): # loop over all TRs
    if np.any(design_concat[TR]): # only get rows where a stimulus was presented
        sentence_id = np.argwhere(design_concat[TR])[0,0]+1
        sent_id_ordered.append(sentence_id)
        
sent_id_ordered = np.array(sent_id_ordered)
print('Sent IDs in order of presentation:',sent_id_ordered)

# Get indices of sentence repeats
repindices = []
for sent_id in range(design_concat.shape[1]): # loop over every sentence id
    # find indices of send_id_order were each sentence was shown
    index_of_send_id_order = np.argwhere(sent_id_ordered==(sent_id+1))[:,0] 
    if len(index_of_send_id_order) >= 1:
        repindices.append(index_of_send_id_order)

repindices = np.vstack(np.array(repindices)).T
num_repeats = np.shape(repindices)[0]
print("There are "+str(np.shape(repindices)[0])+" repetitions of "+str(np.shape(repindices)[1])+" unique sentences")



### ------------------------------------------Perform GLMSingle Regression------------------------------------------

## Specify parameters for GLMsingle run

# Load characters displayed at each TR
chars_by_TRs = np.loadtxt(input_dir+'sub-'+sub_id+'_chars_by_TR.txt', delimiter=' ', dtype='int', encoding='utf-8')
print('Chars each TR matrix:',chars_by_TRs.shape)

# Load button press indices
left_button_location = input_dir+'sub-'+sub_id+'_left_press_by_TR.txt'
left_press_by_TR = np.loadtxt(left_button_location, delimiter=' ', dtype='int', encoding='utf-8')
right_button_location = input_dir+'sub-'+sub_id+'_right_press_by_TR.txt'
right_press_by_TR = np.loadtxt(right_button_location, delimiter=' ', dtype='int', encoding='utf-8')
print('Left presses each TR matrix:',left_press_by_TR.shape)
print('Right presses each TR matrix:',right_press_by_TR.shape)

all_regressors = []
for run in np.arange(num_runs):
    run_data = [left_press_by_TR[run], right_press_by_TR[run], chars_by_TRs[run]]
    all_regressors.append(np.array(run_data).T)
print('Num regressors',len(all_regressors),'x',all_regressors[0].shape)

# Define key options
opt = dict()
opt['wantlibrary'] = 1  # determine the best HRF for each voxel using the library-of-HRFs approach
opt['wantglmdenoise'] = 1  # perform GLMdenoise
opt['wantfracridge'] = 1  # perform ridge regression
opt['wantfileoutputs'] = [1,0,0,1]
opt['wantmemoryoutputs'] = [0,0,0,0]
opt['chunklen'] = 200000 # number of voxels processed at once
opt['extra_regressors'] = all_regressors

# Create GLM_Single object
glmsingle_obj = GLM_single(opt)
print('Created GLMsingle object')

# Visualize all the hyperparameters
pprint(glmsingle_obj.params)



## Run GLMsingle analysis
if run_full:
    start_time = time.time()
    print('Start:',start_time)
    
    print(f'running GLMsingle...')
    
    # run GLMsingle
    results_glmsingle = glmsingle_obj.fit(
    design,
    data,
    stim_duration,
    tr,
    outputdir=output_dir,
    figuredir=None)

    elapsed_time = time.time() - start_time

    print('Finished after: ', f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
    


## Load raw GLMsingle file outputs
results_glmsingle = dict()
results_glmsingle['typea'] = np.load(join(output_dir,'TYPEA_ONOFF.npy'),allow_pickle=True).item()
results_glmsingle['typed'] = np.load(join(output_dir,'TYPED_FITHRF_GLMDENOISE_RR.npy'),allow_pickle=True).item()
betas = results_glmsingle['typed']['betasmd'] # get trial-wise betas from raw output of GLMsingle
onoffs = results_glmsingle['typea']['onoffR2']
print('\nbetas shape:',betas.shape)
print('on-off shape:',onoffs.shape)


### ------------------------------------------Extract and Save Results to Files------------------------------------------

## Extract and store betas in the correct order
condition_list = np.unique(sent_id_ordered)

# Compute average betas over repeats of conditions in the right order
betas_ordered_all = np.zeros([np.shape(betas)[0],np.shape(betas)[1],np.shape(betas)[2],len(condition_list),num_repeats])
betas_ordered_mean = np.zeros([np.shape(betas)[0],np.shape(betas)[1],np.shape(betas)[2],len(condition_list)])
betas_ordered_tstat = np.zeros([np.shape(betas)[0],np.shape(betas)[1],np.shape(betas)[2],len(condition_list)])
for k in condition_list:
    indices = np.where(sent_id_ordered==k)[0]
    betas_ordered_all[:,:,:,k-1,:] = betas[:,:,:,indices]
    betas_ordered_mean[:,:,:,k-1] = np.mean(betas[:,:,:,indices],3)  # axis 3 is the final axis
    betas_ordered_tstat[:,:,:,k-1] = np.mean(betas[:,:,:,indices],3)/np.std(betas[:,:,:,indices],3)

# Print shapes
print('all betas:',betas_ordered_all.shape)
print('mean betas:',betas_ordered_mean.shape)
print('t-stat betas:',betas_ordered_tstat.shape)
print('')


## Save .nii files

# Load the BOLD data to get affine transformation
func = nibabel.load(os.path.join(input_dir,datafiles[0]))

# Save mean betas
ni_img = nibabel.Nifti1Image(betas_ordered_mean, func.affine)
betas_filename = output_dir+'/sub-'+sub_id+'_GLMSbetas-mean.nii.gz'
nibabel.save(ni_img, betas_filename)

# Save single betas
for repeat in np.arange(num_repeats):
    ni_img = nibabel.Nifti1Image(betas_ordered_all[:,:,:,:,repeat], func.affine)
    repeat_filename = output_dir+'/sub-'+sub_id+'_GLMSbetas-rep-'+str(repeat)+'.nii.gz'
    nibabel.save(ni_img, repeat_filename)

# Save tstats
ni_img = nibabel.Nifti1Image(betas_ordered_tstat, func.affine)
tstats_filename = output_dir+'/sub-'+sub_id+'_GLMSbetas-tstat.nii.gz'
nibabel.save(ni_img, tstats_filename)

# Save onoffs
ni_img = nibabel.Nifti1Image(onoffs, func.affine)
onoff_filename = output_dir+'/sub-'+sub_id+'_GLMSbetas-onoff.nii.gz'
nibabel.save(ni_img, onoff_filename)
print('Saved .nii files')



## Save .csv files

# Load combined mask
# mask_filename = input_dir+'/sub-'+sub_id+'_combined_mask.nii'

# # Save mean betas
# betas_ordered_mean_flat = apply_mask(betas_filename, mask_filename)
# np.savetxt(output_dir+'/sub-'+sub_id+'_GLMSbetas-mean.csv', betas_ordered_mean_flat, delimiter=',', fmt='%.3f')

# # Save single betas
# betas_ordered_flat_all = []
# for repeat in np.arange(num_repeats):
#     repeat_filename = output_dir+'/sub-'+sub_id+'_GLMSbetas-rep-'+str(repeat)+'.nii.gz'
#     betas_ordered_flat = apply_mask(repeat_filename, mask_filename)
#     save_name = output_dir+'/sub-'+sub_id+'_GLMSbetas-rep-'+str(repeat)+'.csv'
#     np.savetxt(save_name, betas_ordered_flat, delimiter=',', fmt='%.3f')
#     betas_ordered_flat_all.append(betas_ordered_flat)
# betas_ordered_flat_all = np.array(betas_ordered_flat_all)
# print('Flat betas dim:',betas_ordered_flat_all.shape)

# # Save tstats
# betas_ordered_tstat_flat = apply_mask(tstats_filename, mask_filename)
# np.savetxt(output_dir+'/sub-'+sub_id+'_GLMSbetas-tstat.csv', betas_ordered_tstat_flat, delimiter=',', fmt='%.3f')

# # Save onoffs
# onoffs_flat = apply_mask(onoff_filename, mask_filename)
# np.savetxt(output_dir+'/sub-'+sub_id+'_GLMSbetas-onoff.csv', onoffs_flat, delimiter=',', fmt='%.3f')
# print('Saved .csv files')

# # Save ROI values
# ROI_filename = 'ROI_language_SN220_MNI152_resampled-Fodor2024-flip.nii'
# ROI_values = apply_mask(ROI_filename, mask_filename)
# np.savetxt(output_dir+'/sub-'+sub_id+'_ROI-LanSN220.csv', ROI_values, delimiter=',', fmt='%d')

# # Save Glasser Atlas values
# ROI_filename = 'MNI_Glasser_HCP_v1.0_resampled-Fodor2024.nii'
# ROI_values = apply_mask(ROI_filename, mask_filename)
# np.savetxt(output_dir+'/sub-'+sub_id+'_ROI-Glasser.csv', ROI_values, delimiter=',', fmt='%d')



## Compute and save stability scores

# Define key variables
run_num_set = np.arange(num_repeats) # pairs of runs start every second run
run_combins = list(itertools.combinations(run_num_set, 2)) # get all combinations of run pairs
print('Number of combinations for stability scores:', len(run_combins))

# Loop over all voxels
stability_scores_all = np.zeros([np.shape(betas)[0],np.shape(betas)[1],np.shape(betas)[2]])
variance_scores_all = np.zeros([np.shape(betas)[0],np.shape(betas)[1],np.shape(betas)[2]])
for i in np.arange(stability_scores_all.shape[0]):
    for j in np.arange(stability_scores_all.shape[1]):
        for k in np.arange(stability_scores_all.shape[2]):

            # compute correlation between presentations of same stimulus (stability score)
            stability_scores = []
            for run_combin in run_combins:        
                voxel_activity_1 = betas_ordered_all[i,j,k,:,run_combin[0]]
                voxel_activity_2 = betas_ordered_all[i,j,k,:,run_combin[1]]
                stability_scores.append(np.corrcoef(voxel_activity_1,voxel_activity_2)[0,1])
            stability_score = np.mean(stability_scores)

            # compute variance score (variation between conditions divide by all variation)
            voxel_all_variation = np.var(betas_ordered_all[i,j,k,:,:])
            voxel_between_variation = np.var(betas_ordered_mean[i,j,k,:])
            variance_score = (voxel_between_variation/voxel_all_variation)

            # save stability and variance scores
            stability_scores_all[i,j,k] = stability_score
            variance_scores_all[i,j,k] = variance_score

print('Stability score:',stability_scores_all.shape)
print('Variance score:',variance_scores_all.shape)

# Save stability scores
ni_img = nibabel.Nifti1Image(stability_scores_all, func.affine)
betas_filename = output_dir+'/sub-'+sub_id+'_GLMSbetas-stabilities.nii.gz'
nibabel.save(ni_img, betas_filename)

# Save variance scores
ni_img = nibabel.Nifti1Image(variance_scores_all, func.affine)
betas_filename = output_dir+'/sub-'+sub_id+'_GLMSbetas-variances.nii.gz'
nibabel.save(ni_img, betas_filename)


## Produce masks to use with .nii files

# Function to produce binary mask from an existing non-binary mask or atlas
def binarise_mask(mask_path, output_name, threshold, smooth=False):
    # Load original mask and smooth
    mask_obj = nibabel.load(mask_path)
    if smooth==True:
        mask_obj_smooth = nilearn.image.smooth_img(mask_obj, fwhm=4.8)
    else:
        mask_obj_smooth = mask_obj
    mask_array = np.array(mask_obj_smooth.dataobj)

    # Construct new binary mask
    binary_mask_array = np.zeros([np.shape(mask_array)[0],np.shape(mask_array)[1],np.shape(mask_array)[2]])
    for i in np.arange(mask_array.shape[0]):
        for j in np.arange(mask_array.shape[1]):
            for k in np.arange(mask_array.shape[2]):
                if mask_array[i,j,k] > threshold:
                    binary_mask_array[i,j,k] = 1
                else:
                    binary_mask_array[i,j,k] = 0
    
    # Save new binary mask
    print('num voxels in mask:',int(np.sum(binary_mask_array)))
    binary_mask_obj = nibabel.Nifti1Image(binary_mask_array, nibabel.load(mask_path).affine)
    nibabel.save(binary_mask_obj, output_name)


# whole brain mask
mask_input_path = input_dir+'/sub-'+sub_id+'_combined_mask.nii.gz'
mask_output_path = output_dir+'/sub-'+sub_id+'_mask-wholebrain.nii.gz'
all_voxel_data = nibabel.load(mask_input_path)
all_voxel_data_np = np.array(all_voxel_data.dataobj)
print('saving whole brain mask')
print('num voxels in mask:',int(np.sum(all_voxel_data_np)))
shutil.copy(mask_input_path, mask_output_path)

# acunzo mask
mask_input_path = "../Data-fmri/atlases/Acunzo_2022_0.3_mean_resampled-Fodor2024-mask.nii.gz"
mask_output_path = output_dir+'/sub-'+sub_id+'_mask-acunzo.nii.gz'
all_voxel_data = nibabel.load(mask_input_path)
all_voxel_data_np = np.array(all_voxel_data.dataobj)
print('saving acunzo mask')
print('num voxels in mask:',int(np.sum(all_voxel_data_np)))
shutil.copy(mask_input_path, mask_output_path)

# language ROI mask
mask_input_path = "../Data-fmri/atlases/ROI_language_SN220_MNI_resampled-Fodor2024-flip-mask.nii.gz"
mask_output_path = output_dir+'/sub-'+sub_id+'_mask-lanROI.nii.gz'
all_voxel_data = nibabel.load(mask_input_path)
all_voxel_data_np = np.array(all_voxel_data.dataobj)
print('saving lanROI mask')
print('num voxels in mask:',int(np.sum(all_voxel_data_np)))
shutil.copy(mask_input_path, mask_output_path)

# fMRIprep cortical mask
mask_input_path = input_dir+'/sub-'+sub_id+'_cortex_prob.nii.gz'
mask_output_path = output_dir+'/sub-'+sub_id+'_mask-cortex.nii.gz'
print('saving cortical mask')
binarise_mask(mask_input_path, mask_output_path, threshold=0, smooth=False)

# Glasser cortical mask
mask_input_path = "../Data-fmri/atlases/MNI_Glasser_HCP_v1.0_resampled-Fodor2024-cortex-mask.nii.gz"
mask_output_path = output_dir+'/sub-'+sub_id+'_mask-GlasserCortex.nii.gz'
all_voxel_data = nibabel.load(mask_input_path)
all_voxel_data_np = np.array(all_voxel_data.dataobj)
print('saving GlasserCortex mask')
print('num voxels in mask:',int(np.sum(all_voxel_data_np)))
shutil.copy(mask_input_path, mask_output_path)

# Glasser cortical mask (no visual cortex: V1,V2,V3,V4)
mask_input_path = "../Data-fmri/atlases/MNI_Glasser_HCP_v1.0_resampled-Fodor2024-novis-mask.nii.gz"
mask_output_path = output_dir+'/sub-'+sub_id+'_mask-GlasserNovis.nii.gz'
all_voxel_data = nibabel.load(mask_input_path)
all_voxel_data_np = np.array(all_voxel_data.dataobj)
print('saving GlasserNovis mask')
print('num voxels in mask:',int(np.sum(all_voxel_data_np)))
shutil.copy(mask_input_path, mask_output_path)

# Glasser cortical mask (no visual cortex extended: V1,V2,V3,V4,V6,V7,V8,V3A,V3B,FFA)
mask_input_path = "../Data-fmri/atlases/MNI_Glasser_HCP_v1.0_resampled-Fodor2024-novisE-mask.nii.gz"
mask_output_path = output_dir+'/sub-'+sub_id+'_mask-GlasserNovisE.nii.gz'
all_voxel_data = nibabel.load(mask_input_path)
all_voxel_data_np = np.array(all_voxel_data.dataobj)
print('saving GlasserNovisE mask')
print('num voxels in mask:',int(np.sum(all_voxel_data_np)))
shutil.copy(mask_input_path, mask_output_path)

# fMRI average lanROI-stable min&diff diags cortical mask
mask_input_path = "../Data-fmri/atlases/MNI_Glasser_HCP_v1.0_resampled-Fodor2024-cortex-mask.nii.gz"
mask_output_path = output_dir+'/sub-'+sub_id+'_mask-GlasserCortex.nii.gz'
all_voxel_data = nibabel.load(mask_input_path)
all_voxel_data_np = np.array(all_voxel_data.dataobj)
print('saving fMRIAv mask')
print('num voxels in mask:',int(np.sum(all_voxel_data_np)))
shutil.copy(mask_input_path, mask_output_path)

