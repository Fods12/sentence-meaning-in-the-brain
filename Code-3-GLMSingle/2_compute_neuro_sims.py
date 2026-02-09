## Code to extract pairwise similarities between sentences across voxels for different masks and stability thresholds
# Must have previously run or downloaded the GLMSingle betas and stabilities data for each subject (see 1_compute_GLMsingle_Fodor2024.py)
# Script adapted from https://github.com/cvnlab/GLMsingle/tree/main/examples
# James Fodor 2024

# Load and import packages
import numpy as np
import itertools
import warnings
import sys
import os
import nibabel
import nilearn

from nilearn import masking
from nilearn import image

# Remove annoying warnings
warnings.filterwarnings('ignore')


# Produce set of pairwise similarities for a given combined mask
def save_masked_similarities(neuro_data, primary_mask, secondary_mask, sent_id_pairs, save_file_base, smooth=False):
    # Combine primary and secondary mask together
    # See https://nilearn.github.io/dev/modules/generated/nilearn.masking.intersect_masks.html#nilearn.masking.intersect_masks
    combine_mask_obj = masking.intersect_masks([primary_mask, secondary_mask], threshold=1, connected=False)

    # Apply combined mask to the GLMSbetas data
    if smooth==True:
        neuro_data = image.smooth_img(neuro_data, fwhm=4.8)
    masked_data_raw = masking.apply_mask(neuro_data, combine_mask_obj).T

    # Normalise data across activity of each voxel
    masked_data_norm = (masked_data_raw - np.mean(masked_data_raw, axis=0)) / (np.std(masked_data_raw, axis=0)+0.001)
    print('voxels:',masked_data_norm.shape[0])

    # Compute pairwise similarities with combined mask
    neuro_sims_raw = simple_pairwise_sims(masked_data_raw, sent_id_pairs) # compute and save cosine sims
    neuro_sims_norm = simple_pairwise_sims(masked_data_norm, sent_id_pairs)

    # Save similarity results to file
    np.savetxt(save_file_base+'-raw.txt', neuro_sims_raw, fmt='%.4f')
    np.savetxt(save_file_base+'-norm.txt', neuro_sims_norm, fmt='%.4f')


# Function to compute pairwise stimulus similarities across voxels
def simple_pairwise_sims(data_all, sent_id_pairs):
    neuro_sim_storage = np.array([]) # storage array
    for sent_id_pair in sent_id_pairs:
        pair_sim = cosine_sim(data_all[:,sent_id_pair[0]], data_all[:,sent_id_pair[1]])
        neuro_sim_storage = np.append(neuro_sim_storage,pair_sim)
    return neuro_sim_storage

# Calculate cosine similarity between two embeddings
def cosine_sim(embed_1: list, embed_2: list):
    if np.dot(embed_1,embed_2) == 0:
        similarity = 0 # don't normalise if similarity is zero
    else:
        similarity = np.dot(embed_1,embed_2)/(np.linalg.norm(embed_1)*np.linalg.norm(embed_2))
    return similarity

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


# Load stimuli set
stimuli_path = '../Fodor2024-final108_sentences.txt'
all_sentences = np.loadtxt(stimuli_path, delimiter=',', dtype='str', encoding='utf-8')
all_sentences_indices = np.arange(len(all_sentences))
sent_id_pairs = list(itertools.combinations(all_sentences_indices, 2))
sent_id_indices = np.arange(len(list(itertools.combinations(all_sentences_indices, 2))))

# Get command line arguments
sbj_id = str(sys.argv[1])  # subject ID
av_type = int(sys.argv[2])  # how to average over repeats (0 = mean, 1 = tstat)

# Path to the GLMSingle betas data
file_prefix = '../Data-fmri/preprocessed/sub-'+sbj_id+'-nii/sub-'+sbj_id

# Make stablities raw mask
stability_thresholds = [0.04,0.07,0.10]
print('Mean version')
for stability_threshold in stability_thresholds:
    mask_input_path = file_prefix+'_GLMSbetas-stabilities.nii.gz'
    mask_output_path = file_prefix+'_mask-stabilities-mean-'+str(stability_threshold)+'.nii.gz'
    print('saving '+str(stability_threshold)+' raw stabilties mask')
    binarise_mask(mask_input_path, mask_output_path, threshold=stability_threshold, smooth=False)

# Loop over stability thresholds and masks and save data for each
masks_dir = '../Data-fmri/atlases/'
masks = [f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')]
masks.append('wholebrain.nii.gz') # add whole brain mask for this participant
masks.append('cortex.nii.gz') # add cortex mask for this participant
big_masks = ['wholebrain','cortex','Acunzo-02','Acunzo-03','Acunzo-04','fMRIAv-04','fMRIAv-05',
             'Glasser-cortex','Glasser-cortex_novis','Glasser-cortex_novisE','LanROI-all']

for stability_threshold in stability_thresholds:

    for mask in masks:

        # skip the 0.15 stability threshold for masks with too few voxels
        if stability_threshold==0.10 and mask[:-7] not in big_masks:
            continue

        # choose correct folder prefix
        if mask in ('wholebrain.nii.gz', 'cortex.nii.gz'):
            primary_mask = file_prefix+'_mask-'+mask # mask from individual subject folder
        else:
            primary_mask = masks_dir+mask # masks from the 'masks' folder
        
        # define main masks to use
        whole_brain_mask = file_prefix+'_mask-wholebrain.nii.gz'
        output_dir = 'sub-'+sbj_id+'-sims'

        # different stability masks depending on whether we use mean or tstat method
        if av_type == 0:
            neuro_data = file_prefix+'_GLMSbetas-mean.nii.gz'
            save_name = output_dir+'/sub-'+sbj_id+'_similarities-mean-'+mask[:-7]
            secondary_mask = file_prefix+'_mask-stabilities-mean-'+str(stability_threshold)+'.nii.gz'
        elif av_type == 1:
            neuro_data = file_prefix+'_GLMSbetas-tstat.nii.gz'
            save_name = output_dir+'/sub-'+sbj_id+'_similarities-tstat-'+mask[:-7]
            secondary_mask = file_prefix+'_mask-stabilities-mean-'+str(stability_threshold)+'.nii.gz'

        # create output directory if needed
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # save results
        print(mask+' all')
        save_masked_similarities(neuro_data, primary_mask, whole_brain_mask, sent_id_pairs, save_name, smooth=False)
        print(mask+' stable-'+str(stability_threshold))
        save_masked_similarities(neuro_data, primary_mask, secondary_mask, sent_id_pairs, save_name+'-stable-'+str(stability_threshold), smooth=False)
    
