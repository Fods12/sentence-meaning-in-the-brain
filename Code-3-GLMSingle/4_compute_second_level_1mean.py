## Code to run a second order GLM model on searchlight-RSA data for all subjects
# Adapted from: https://nilearn-doc-dev.github.io/auto_examples/05_glm_second_level/plot_second_level_one_sample_test.html
# See documentation here: https://nilearn.github.io/dev/modules/generated/nilearn.glm.second_level.non_parametric_inference.html
# James Fodor 2024

import pandas as pd
import glob
import nibabel
import numpy as np
import warnings
import re

from nilearn import image
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import non_parametric_inference

# Mute UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Specify set of all models to compute for
model_storage_path = '../Data-models/similarities/'
model_paths = glob.glob(f'{model_storage_path}/*.txt') # loads all models in the folder - replace with subset if desired
regex_pattern = r"Fodor2024-final108_neuro_(.*?)\.txt" # get the model name from the file name using regex
model_names = [re.search(regex_pattern, string).group(1) for string in model_paths]
stim_set = 'diags' #diags, all

# Loop over all models
for idx,model_name in enumerate(model_names):

    # Load input nii data for all subjects
    input_nii_data = glob.glob(f'*/sub-*'+model_name+'_set-'+stim_set+'.nii.gz')
    print(model_name+'_set-'+stim_set+'.nii.gz')
    nii_images = [image.load_img(nii_file) for nii_file in input_nii_data]

    # Create binary mask where std > 0 and only in cortical mask region
    concat_images = image.concat_imgs(nii_images)
    cortical_mask = image.load_img("../Data-fmri/atlases/Glasser-cortex.nii.gz")
    std_images = image.math_img("np.std(img, axis=-1)", img=concat_images)
    binary_mask = image.math_img("(img > 0)*mask", img=std_images, mask=cortical_mask)
    # binary_mask.to_filename('second_level_results/'+model_name+'_mask.nii.gz')

    # Set parameters
    run_tcfe = True
    GLM_fwhm = 8 # in mm

    # Specify second-order GLM
    print('Running '+model_name)
    design_matrix = pd.DataFrame([1] * len(input_nii_data), columns=['intercept']) # single sample t-test
    second_level_model = SecondLevelModel(smoothing_fwhm=GLM_fwhm)
    second_level_model = second_level_model.fit(input_nii_data, design_matrix=design_matrix)
    model_outputs = non_parametric_inference(input_nii_data, design_matrix=design_matrix, 
                        model_intercept=True,
                        mask=binary_mask,
                        n_perm=100000, # number permuations for non-parametric p-value correction 
                        two_sided_test=False, # False means positive effects only
                        smoothing_fwhm=GLM_fwhm,
                        n_jobs=-1, # CPUs to use
                        threshold=0.001, # for cluster-based inference
                        tfce=run_tcfe) # perform threshold-free cluster enhancement inference

    # Save output to .nii file
    # nibabel.save(model_outputs['logp_max_t'], 'second_level_results/'+model_name+'_logp-t.nii.gz')
    # nibabel.save(model_outputs['logp_max_mass'], 'second_level_results/'+model_name+'_logp-mass.nii.gz')
    if run_tcfe:
        nibabel.save(model_outputs['logp_max_tfce'], 'second_level_results/'+model_name+'_set-'+stim_set+'_size-'+str(GLM_fwhm)+'_logp-tfce-big.nii.gz')
    print('Saved '+model_name)