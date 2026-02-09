# Script to extract .nii and .json files from the original folder structure and move them to a new folder structure that follows BIDS format. 
# It also extracts .dcm physio files and moves them to a new folder. 

import os
import sys
import re
import shutil
import zipfile
import json
import numpy as np
  

def move_and_rename_files(input_directory, output_directory, subject_id):
    # Get a list of all subdirectories in the main folder
    subdirectories = [subdir for subdir in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, subdir))]
    
    for sub_dir in ['/func', '/anat', '/fmap']: # Create new subdirectories for BIDS format
        if not os.path.exists(output_directory+sub_dir):
            os.makedirs(output_directory+sub_dir)

    for subdir in subdirectories:
        # Check if the subdirectory name contains the pattern 'RunN' where N is a digit
        run_match = re.search(r'Run(\d+)', subdir)
        structural_match = ('MP2RAGE' in subdir) and ('UNI-DEN' in subdir) and ('UNI-DEN_ND' not in subdir)
        reverse_phase_match = 'ReversePhase' in subdir and 'SBRef' not in subdir
        
        # Get functional files
        if run_match:
            # Extract the digit N from the matched pattern
            run_number = run_match.group(1)

            # Iterate over files in the subdirectory with '.nii' extension
            for foldername, _, filenames in os.walk(input_directory+'/'+subdir):
                # BOLD or SBref
                if subdir.split('_')[-1] == 'cmrr':
                    file_type = 'bold'
                elif subdir.split('_')[-1] == 'SBRef':
                    file_type = 'sbref'
                
                for filename in filenames:
                    if filename.endswith('.nii'):
                        
                        # Build the new filename as 'RunN.nii'
                        new_filename = f'sub-'+subject_id+'_task-sentences_run-'+run_number+'_'+file_type+'.nii'

                        # Construct the full paths for the source and destination files
                        source_path = os.path.join(foldername, filename)
                        destination_path = os.path.join(output_directory+'/func', new_filename)

                        # Move and rename the file
                        shutil.move(source_path, destination_path)
                        print(f"Moved {filename} to {new_filename}")
                        
                    elif filename.endswith('.json'):
                        
                        # Build the new filename as 'RunN.nii'
                        new_filename = f'sub-'+subject_id+'_task-sentences_run-'+run_number+'_'+file_type+'.json'

                        # Construct the full paths for the source and destination files
                        source_path = os.path.join(foldername, filename)
                        destination_path = os.path.join(output_directory+'/func', new_filename)

                        # Move and rename the file
                        shutil.move(source_path, destination_path)
                        
                        # Add the task name to the json file
                        add_entry_to_json(destination_path, 'TaskName', 'Sentences')
    
        # Get structural files
        elif structural_match:
            # For subdirectories containing 'MP2RAGE'
            mp2rage_path = os.path.join(input_directory, subdir)

            # Traverse all nested directories within the 'MP2RAGE' subdirectory
            for foldername, _, filenames in os.walk(mp2rage_path):
                for filename in filenames:
                    # Check if the file ends with '.nii'
                    if filename.endswith('.nii'):
                        file_path = os.path.join(foldername, filename)

                        # Copy and rename the file to the main folder
                        new_filename = 'sub-'+subject_id+'_T1w.nii'
                        new_path = os.path.join(output_directory+'/anat', new_filename)
                        shutil.move(file_path, new_path)
                        print(f'Copied: {file_path} to {new_path}')
                        
                    elif filename.endswith('.json'):
                        file_path = os.path.join(foldername, filename)

                        # Copy and rename the file to the main folder
                        new_filename = 'sub-'+subject_id+'_T1w.json'
                        new_path = os.path.join(output_directory+'/anat', new_filename)
                        shutil.move(file_path, new_path)
                        print(f'Copied: {file_path} to {new_path}')
        
        # Get reverse phase file
        elif reverse_phase_match:
            
            # For subdirectories containing 'ReversePhase'
            reverse_phase_path = os.path.join(input_directory, subdir)

            # Traverse all nested directories within the 'ReversePhase' subdirectory
            for foldername, _, filenames in os.walk(reverse_phase_path):
                for filename in filenames:
                    # Check if the file ends with '.nii'
                    if filename.endswith('.nii'):
                        file_path = os.path.join(foldername, filename)

                        # Move and rename the file to the main folder
                        new_filename = 'sub-'+subject_id+'_acq-case4_dir-LR_epi.nii'
                        new_path = os.path.join(output_directory+'/fmap', new_filename)
                        shutil.move(file_path, new_path)
                        print(f'Moved: {file_path} to {new_path}')
                        
                    elif filename.endswith('.json'):
                        file_path = os.path.join(foldername, filename)

                        # Copy and rename the file to the main folder
                        new_filename ='sub-'+subject_id+'_acq-case4_dir-LR_epi.json'
                        new_path = os.path.join(output_directory+'/fmap', new_filename)
                        shutil.move(file_path, new_path)
                        print(f'Copied: {file_path} to {new_path}')
                        
                        # Add func runs to use the reverse phase for to json file
                        reverse_phase_runs = ["func/sub-"+subject_id+"_task-sentences_run-"+str(run_id)+"_bold.nii" for run_id in np.arange(1,9)]
                        add_entry_to_json(new_path, 'IntendedFor', reverse_phase_runs)
                        
                    
# Unzip compressed folders
def extract_archives(main_folder):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_path = os.path.splitext(zip_path)[0]
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                os.remove(zip_path)  # Delete the archive file after extraction
                    

# Extract and rename .dcm physio files
def find_and_copy_dcm_files(main_folder, subject_id):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".dcm"):
                dcm_path = os.path.join(root, file)
                # Use the entire existing path as the new filename
                full_path_name = os.path.relpath(dcm_path, main_folder).replace(os.path.sep, '_')
                run_match = re.search(r'Run(\d+)', full_path_name)
                run_number = run_match.group(1)
                new_filename = 'sub-'+subject_id+'_physio_run'+str(run_number)+'.dcm'
                destination_path = os.path.join(main_folder+'/phys', new_filename)
                shutil.copy(dcm_path, destination_path)
                print(f'Moved: {dcm_path} to {destination_path}')
                
                
# Append a new item to a json file
def add_entry_to_json(filename, new_key, new_value):
    new_entry = {new_key: new_value}

    with open(filename) as f:
        data = json.load(f)

    data.update(new_entry)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=0)


# Extract files
subject_id = sys.argv[1] # dicoms number participants from 1 rather than 0 so their subject IDs are +1 above mine
subject_id_dicom = str(int(subject_id)+1)
intput_folder = '/data/fMRIprep_input/1.7.167/1.7.167.'+subject_id_dicom+'/1.7.167.'+subject_id_dicom+'.1/1.7.167.'+subject_id_dicom+'.1.1/nifti_series/'
output_folder = '/data/fMRIprep_input/sub-'+sys.argv[1]+'/'
move_and_rename_files(intput_folder, output_folder, subject_id)