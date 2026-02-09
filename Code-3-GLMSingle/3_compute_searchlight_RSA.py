## Compute voxelwise searchlight RSA using results from GLMSingle
# Adapted from https://users.aalto.fi/~vanvlm1/mne-rsa/functions/mne_rsa.rsa_nifti.html
# James Fodor 2024

import mne_rsa
import os
import sys
import nibabel
import nilearn
import numpy as np
import glob
import re

def mask_based_utri2mat(upper_triangle, ntotal):
    """ Convert a flattened upper trianngular matrix into an nxn matrix.

    Args:
        upper_triangle (np.ndarray): a flattened upper triangular matrix
        ntotal (int): the dimension of the full matrix

    Returns:
        np.ndarray: full nxn matrix
    """
    out = np.ones((ntotal,ntotal)) # output array
    mask = np.triu(np.ones((ntotal,ntotal),dtype=bool), k=1) # create mask for upper triangle
    out[mask] = upper_triangle  # upper triangular elements with mask
    out.T[mask] = upper_triangle  # lower triangular elements with transposed mask
    return out 


# Path to the neuro dataset
sbj_id = str(sys.argv[1])  # subject ID
radius = int(sys.argv[2])  # spatial radius of searchlight in mm
voxel_set = str(sys.argv[3])  # use all sentence pairs or just diag ones
file_prefix = '../Data-fmri/betas/sub-'+sbj_id+'-nii/sub-'+sbj_id
neuro_data = file_prefix+'_GLMSbetas-mean.nii.gz'

# Load and normalise beta coefficients
all_voxel_path= nibabel.load(neuro_data)
all_voxel_data = all_voxel_path.get_fdata()
all_voxel_data_norm = (all_voxel_data - np.mean(all_voxel_data, axis=0)) / (np.std(all_voxel_data, axis=0)+0.001)
all_voxel_data_norm_nifti = nibabel.Nifti1Image(all_voxel_data_norm, all_voxel_path.affine, all_voxel_path.header)

# Load masks
mask_all_brain = nibabel.load(file_prefix+'_mask-wholebrain.nii.gz')
mask_cortex = nibabel.load(file_prefix+'_mask-cortex.nii.gz')

# Load model similarities (note: code requires a list of more than one model)
model_storage_path = '../Data-models/similarities/'
model_paths = glob.glob(f'{model_storage_path}/*.txt') # loads all models in the folder - replace with subset if desired
model_sims_storage_all = []
model_sims_storage_diags = []
sent_pair_ids_diag = np.loadtxt('./diag_sent_ids.txt')
for model_path in model_paths:
    with open(model_path, encoding='utf-8') as file:
        model_sims = np.array([float(line.rstrip('\n')) for line in file]) # load one line at a time
    model_sims_diags = np.array([model_sims[idx] if idx in sent_pair_ids_diag else np.NaN for idx in np.arange(len(model_sims))])
    model_sims_storage_all.append(1-model_sims)
    model_sims_storage_diags.append(1-model_sims_diags)

# Perform searchlight rsa for all models
if voxel_set=='all':
    searchlight_rsas = mne_rsa.rsa_nifti(all_voxel_data_norm_nifti, model_sims_storage_all, brain_mask=mask_cortex, \
                       spatial_radius=float(radius/1000), image_rdm_metric='cosine', ignore_nan=True, n_jobs=32)
elif voxel_set=='diags':
    searchlight_rsas_diags_only = mne_rsa.rsa_nifti(all_voxel_data_norm_nifti, model_sims_storage_diags, brain_mask=mask_cortex, \
                                  spatial_radius=float(radius/1000), image_rdm_metric='cosine', ignore_nan=True, n_jobs=-1)
     
# Save results for each model
regex_pattern = r"Fodor2024-final108_neuro_(.*?)\.txt" # get the model name from the file name using regex
model_names = [re.search(regex_pattern, string).group(1) for string in model_paths]
for idx,model_name in enumerate(model_names):

    if voxel_set=='all':
        searchlight_rsa = searchlight_rsas[idx] # get results for single model
        output_name = file_prefix+'_searchlight-'+str(radius)+'mm_model-'+model_name+'_set-all.nii.gz'
        nibabel.save(searchlight_rsa, output_name) # save single model to .nii
    elif voxel_set=='diags':
        searchlight_rsa_diags_only = searchlight_rsas_diags_only[idx]
        output_name = file_prefix+'_searchlight-'+str(radius)+'mm_model-'+model_name+'_set-diags.nii.gz'
        nibabel.save(searchlight_rsa_diags_only, output_name) # save single model to .nii
