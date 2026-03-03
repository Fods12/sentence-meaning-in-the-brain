## A module with functions for processing fMRI voxel data
## James Fodor 2023

import numpy as np
import seaborn as sns
import itertools
import random as rd
import pandas as pd
import random
import pandas as pd
import matplotlib.pyplot as plt
import nibabel
import os
    
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from pingouin import partial_corr

# List of experimental subject IDs for each experimental dataset
experiment_subjects = {'Alice_neuro':['18','22','23','24','26','28','30','31','35','36','37','39','41','42','43','44','45','47','48','49','50','51'], #21,53
                       'Anderson_neuro':['5049','5051','5146','9322','9324','9362','9379','9511','9637','9655','9701','9726','9742'], #13
                       'Wehbe_neuro':['01','02','03','04','05','06','07','08'],
                       'Zhang_neuro':['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19'],
                       'Pereira243_neuro':['00','02','03','04','07','15'],
                       'Pereira384_neuro':['00','02','04','07','08','09','14','15'],
                       'Fodor2024-final108_neuro':['05','06','07','08','09','10','11','12','13','14','16','17','18','19','20','21','22','23','24','25','26','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44']}

# Cortical atlases for each dataset
atlas_names = {'Alice_neuro':'Glasser', 'Zhang_neuro':'Glasser', 'Pereira243_neuro':'Gorden', 'Pereira384_neuro':'Gorden', 'Anderson_neuro':'Destrieux', 'Wehbe_neuro':'None'}

# Sets of voxels to compare
def get_voxel_set_dict(stability_level):
    """_summary_

    Args:
        stability_level (str): threshold of stable voxels to include
    """
    voxel_set_dict = {'standard': ['Glasser-cortex_novis-stable-'+stability_level+'-norm', 'LanROI-all-stable-'+stability_level+'-norm'],
                    'extended': ['Glasser-cortex_novis-stable-'+stability_level+'-norm', 'fMRIAv-05-stable-'+stability_level+'-norm', 'LanROI-all-stable-'+stability_level+'-norm'],
                    'stability': ['Glasser-cortex_novis-norm','Glasser-cortex_novis-stable-0.04-norm','Glasser-cortex_novis-stable-0.07-norm','Glasser-cortex_novis-stable-0.1-norm'], # for stability comparison
                    'lanROIs': ['LanROI-AntTemp-stable-'+stability_level+'-norm','LanROI-PostTemp-stable-'+stability_level+'-norm','LanROI-AG-stable-'+stability_level+'-norm',
                                'LanROI-MFG-stable-'+stability_level+'-norm','LanROI-IFG-stable-'+stability_level+'-norm','LanROI-IFGorb-stable-'+stability_level+'-norm'], # for lanROI comparison
                    'cortexROIs': ['Glasser-dorsomedial_PFC-stable-'+stability_level+'-norm','Glasser-dorsolateral_PFC-stable-'+stability_level+'-norm',
                                    'Glasser-posterior_cingulate-stable-'+stability_level+'-norm','Glasser-precuneus-stable-'+stability_level+'-norm',
                                    'Glasser-primary_sensory-stable-'+stability_level+'-norm','Glasser-primary_visual-stable-'+stability_level+'-norm','LanROI-all-stable-'+stability_level+'-norm']} # for cortical ROI comparison
    
    return voxel_set_dict


# Get diagonal block indices
def get_pairs_in_diag(stimuli_indices_all, num_sets):
    """ Returns a list of sentence stimuli in the diagonal blocks

    Args:
        stimuli_indices_all (list): list of indices for all stimuli
        num_sets (int): number of diagonal block subsets

    Returns:
        (list): list of indices for diagonal stimuli
    """
    num_sents = stimuli_indices_all.shape[0]
    sents_per_set = int(num_sents/num_sets)

    # Compute and save a set of all sentence pairs on the matrix diagonal
    pairs_storage = []
    for set_num in np.arange(num_sets):
        start_id = set_num*sents_per_set
        end_id = start_id+sents_per_set
        single_set_sents = np.arange(start_id,end_id)
        sent_id_pairs = list(itertools.combinations(single_set_sents, 2))
        pairs_storage.append(sent_id_pairs)
        
    sent_id_pairs_diags = [x for set in pairs_storage for x in set]

    # Get set of all sentence pairs ids
    sent_id_pairs = list(itertools.combinations(stimuli_indices_all, 2))

    # Get the numpy array indices for all sentence pairs on matrix diagonal
    sent_pair_ids = []
    for idx,sent_pair in enumerate(sent_id_pairs):
        if sent_pair in sent_id_pairs_diags:
            sent_pair_ids.append(idx)
    sent_pair_ids = np.array(sent_pair_ids)

    return sent_pair_ids


# Calculate cosine similarity between two embeddings
def cosine_sim(embed_1: list, embed_2: list):
    """ Returns the cosine similarity (-1 to 1) between two embeddings, inputted as vectors.

    Args:
        embed_1 (list): word embedding 1
        embed_2 (list): word embedding 2

    Returns:
        float: cosine similarity
    """
    if np.dot(embed_1,embed_2) == 0:
        similarity = 0 # don't normalise if similarity is zero
    else:
        similarity = np.dot(embed_1,embed_2)/(np.linalg.norm(embed_1)*np.linalg.norm(embed_2))
        #similarity, _ = spearmanr(embed_1, embed_2)
    return similarity


# Function to split stimuli indices into random train/dev/test sets
def train_dev_test_split(stimuli_indexes, dataset_name, train_prop, dev_prop):
    """ Function to divide stimuli indexes into train/dev/test splits.

    Args:
        stimuli_indexes (list): list of ints giving all stimuli indexes
        dataset_name (str): name of fMRI dataset

    Returns:
        tuple: lists of train/dev/test splits
    """
    
    # Shuffle data
    stimuli_indexes_local = stimuli_indexes.copy()
    rd.shuffle(stimuli_indexes_local)

    # Split into train/dev/test
    split_1 = int(train_prop * len(stimuli_indexes_local))
    split_2 = int((train_prop+dev_prop) * len(stimuli_indexes_local))
    train_stimuli = stimuli_indexes_local[:split_1].tolist()
    dev_stimuli = stimuli_indexes_local[split_1:split_2].tolist()
    test_stimuli = stimuli_indexes_local[split_2:].tolist()
    
    return train_stimuli,dev_stimuli,test_stimuli


# Function to load stimuli indices from train/dev/test files
def load_train_dev_test_data(dataset_name):
    """ Load train/dev/test split from files

    Args:
        dataset_name (str): name of fMRI dataset

    Returns:
        tuple: lists of train/dev/test splits
    """
    
    train_stimuli = np.loadtxt("Data\\"+dataset_name+"_train_stimuli.txt", dtype=int)
    # dev_stimuli = np.loadtxt("Data\\"+dataset_name+"_dev_stimuli.txt", dtype=int)
    test_stimuli = np.loadtxt("Data\\"+dataset_name+"_test_stimuli.txt", dtype=int)
    
    return train_stimuli,test_stimuli


## Compute r_scores for all included voxels
def compute_rscore_voxelwise(bold_data, model_data, save_file_prefix, voxels_indices_all, train_stimuli, test_stimuli, num_voxels, ridge_alpha=1, full_random=False):
    """ Computes the rscore for each voxel in the included indices.
    This is a measure of the correlation between the voxel activity and the predicted activity given averaged
    ConceptNet embeddings for the corresponding stimuli at each stimulus time point.

    Args:
        save_file_prefix (str): prefix to use for saving files to store results
        dataset_name (str): name of fMRI dataset to use
        data_sentences_norm (np_array): normalised sentences neuro embeddings data
        voxels_indices_all (list): voxel IDs to compute the r-scores for
        train_stimuli (list): training indices used to train regression model
        test_stimuli (list): test indices also used to evaluate the regression model
        model_name (str): name of the computational model to use for computing r-score
        ridge_alpha (int): the value alpha to use in ridge regression
        full_namdom (bool): whether to shuffle order of stimuli to use as control condition

    Returns:
        np.ndarray: r_scores for each voxel
    """

    # Model features to use for training model
    X = model_data
    
    # Randomise embeddings matrix
    if full_random == True:
        import random
        X_shape = X.shape
        X_flat = X.flatten()
        random.shuffle(X_flat)
        X = X_flat.reshape(X_shape)    

    # Loop over all voxels
    voxel_r_scores = []
    for voxel_id in voxels_indices_all:
        
        # get time series for single voxel
        y = bold_data[:,voxel_id]
        
        # check if the y vector is all zeros
        sum_abs_y = sum([abs(number) for number in y])
            
        if sum_abs_y<0.01: 
            voxel_r_scores.append([voxel_id,0]) # set r_score to 0
        
        # all cases with y vector is not all zeros
        else:
            # fit regression model using train stimuli
            reg_model = Ridge(alpha=ridge_alpha)
            reg_model.fit(X[train_stimuli], y[train_stimuli])

            # evaluate r_scores using test and dev stimuli
            y_hat = reg_model.predict(X)
            
            r_score = np.corrcoef(y[test_stimuli],y_hat[test_stimuli])[0,1]
            voxel_r_scores.append([voxel_id,r_score])

    r_scores = np.array(voxel_r_scores)[:,1]
    # np.savetxt(save_file_prefix+"_rscore"+str(num_voxels)+".txt", r_scores, fmt='%f')
    # print('saved voxel r-scores with alpha='+str(ridge_alpha))
    
    return r_scores


# Function to extract all voxels with an r_score above a specified threshold
def extract_highrscore_voxels(data_file_prefix, voxels_indices_all, r_scores, thresholds):
    """ Determines which voxels have an r-score above a specified threshold

    Args:
        data_file_prefix (str): prefix for the current dataset and subject ID 
        voxels_indices_all (list): list of all voxels
        r_scores (np.ndarray): full set of computed r_scores for each voxel
        thresholds (list): values to use for hte threshold (generally 0.2-0.6)

    Returns:
        dict: indices of voxels with r_score above the threshold
    """
    
    # Dictionary to store all values in
    rscore_voxels = {}
    
    # Loop over all thresholds we want to use
    for threshold in thresholds:
        
        # Convert to dictionary
        r_scores_dict = dict(zip(voxels_indices_all, r_scores))

        # Find significant voxels using r scores
        significant_voxel_indices = []
        for index in voxels_indices_all:
            if r_scores_dict[index]>threshold:
                significant_voxel_indices.append(index)
                
        # Print data shape
        print('Threshold',str(threshold),'has',len(significant_voxel_indices),'significant voxels out of',len(voxels_indices_all))

        # Save voxel ids for later use
        np.savetxt(data_file_prefix+"-"+str(threshold)+"_voxels.txt", significant_voxel_indices, fmt='%i')
        np.savetxt(data_file_prefix+"-all_voxels.txt", voxels_indices_all, fmt='%i')
        
        # Store significant voxel indices by threshold
        rscore_voxels[threshold] = significant_voxel_indices
        
    return rscore_voxels


# Function for adding subplot for heatmap plotting of raw voxel similarities
def add_heatmap_subplot(fig, plot_num, correl_matrix, dataset, sbj_id, subplot, colorbar, cmap="gist_rainbow", min=0, max=1):
    """ Adds a subplot of a heatmap for raw voxel similarity data

    Args:
        fig (plt_fig): the matplotlib figure object to which this subplot will be added
        plot_num (tuple): integers specifying the location of the subplot in the matplotlib style, e.g. (2,2,2)
        correl_matrix (numpy): correlation matrix to plot
        dataset (str): name of the dataset to be plotted
        sbj_id (str): subject ID of the data to be plotted 
        subplot (str): voxel selection method for similarity matrix
        colorbar (bool): indicate whether to display colorbar
        cmap (str): specify which cmap to use for heatmap
        min (int): smallest value to include in heatmap
        max (int): largest value to include in heatmap
    """
    fig_axis = fig.add_subplot(*plot_num) #create figure subplot
    label = dataset+' '+sbj_id+' '+subplot
    fig_axis.title.set_text(label) # subfigure figure title
    
    sns.heatmap(correl_matrix, cmap=cmap, vmin=min, vmax=max, square=True, ax=fig_axis, xticklabels=False, yticklabels=False, cbar=colorbar)
    
    
# Function to load pre-computed model similarities from files
def load_model_sims(model_sims_dir, dataset_name, normalisation, sim_type):
    """ Loads all model similarities for a specified neuro dataset, from files storing pre-computed values

    Args:
        model_sims_dir (str): path to directory containing model similarity files
        stimuli_indices_all (list): a list of integers specifying the stimuli; used for creating blank data if file not found
        normalisation (bool): whether to load normalised or raw similarities (cosine or diemsim only)
        sim_type (str): type of similarity to load for embeddings ('cosine' or 'diemsim')
        
    Returns:
        dict: dictionary of model similarity values
    """
    
    model_sims = {}
    all_filenames = os.listdir(model_sims_dir)
    if normalisation==True:
        find_val = -1
    else:
        find_val = 1
        
    for filename in all_filenames:
        file_sim_type = filename.split('_')[-1].split('.')[0] # get the type of similarity
        if file_sim_type not in ['diemsim','cosine']: # load all other types of similarities
            with open(model_sims_dir+'/'+filename, encoding='utf-8') as file:
                model_name = filename.replace('.txt','').replace(dataset_name+'_neuro_', '')
                model_sims[model_name] = np.array([float(line.rstrip('\n')) for line in file])

        elif (sim_type=='all' or file_sim_type==sim_type) and filename.find('norml')!=find_val: # load only specified similarity types
            with open(model_sims_dir+'/'+filename, encoding='utf-8') as file:
                model_name = filename.replace('.txt','').replace(dataset_name+'_neuro_', '')
                model_sims[model_name] = np.array([float(line.rstrip('\n')) for line in file])

    return model_sims


# Function to normalise the set of similarities for each model by setting the min to zero and max to one
def normalise_similarities(model_sims_storage, percentile, group_norm=0):
    """ Returns a normalised dictionary of model similarities

    Args:
        model_sims_storage (dict): dictionary of unnormalised model similarities
        percentile (float): percentile to exclude when normalising (for when there are outliers)
        group_norm (bool): specify whether normalisation is computed for each model separatley (0) or all together (1)

    Returns:
        dict: dictionary of nnormalised model similarities
    """
    model_sims_storage_norml = {}
    
    if group_norm==0: # normalise each model separately
        for model in model_sims_storage.keys():
            if percentile==False:
                min_value = np.min(model_sims_storage[model])
                range_values = np.ptp(model_sims_storage[model])
            else:
                min_value = np.percentile(model_sims_storage[model], percentile)
                range_values = np.percentile(model_sims_storage[model], 100-percentile)-np.percentile(model_sims_storage[model], percentile)

            if min_value<0:
                model_sims_storage_norml[model] = (model_sims_storage[model]+abs(min_value))/range_values
            else:
                model_sims_storage_norml[model] = (model_sims_storage[model]-abs(min_value))/range_values
   
    elif group_norm==1: # normalise all models together
        group_min_value = np.min([np.min(model_sims_storage[voxel_set]) for voxel_set in model_sims_storage.keys()])
        group_range_value = np.max([np.ptp(model_sims_storage[voxel_set]) for voxel_set in model_sims_storage.keys()])
        for model in model_sims_storage.keys():
            if group_min_value<0:
                model_sims_storage_norml[model] = (model_sims_storage[model]+abs(group_min_value))/group_range_value
            else:
                model_sims_storage_norml[model] = (model_sims_storage[model]-abs(group_min_value))/group_range_value
    
    return model_sims_storage_norml


# Function to normalise the set of similarities for each model by setting the min to zero and max to one
def normalise_similarities_single(model_sims_storage):
    """ Returns a normalised dictionary of model similarities

    Args:
        model_sims_storage (dict): dictionary of unnormalised model similarities

    Returns:
        dict: dictionary of nnormalised model similarities
    """
    min_value = np.min(model_sims_storage)
    range_values = np.ptp(model_sims_storage)
    if min_value<0:
        model_sims_storage_norml = (model_sims_storage+abs(min_value))/range_values
    else:
        model_sims_storage_norml = (model_sims_storage-abs(min_value))/range_values
    
    return model_sims_storage_norml


    
# Function to compute pairwise stimulus similarities across voxels
def compute_neuro_pairwise_sims(voxel_indices_for_calc, data_all, stimuli_indices_all, stimuli_indices_compute):
    """ Compute the cosine similarity between all sentences over specified voxels

    Args:
        voxel_indices_for_calc (list): list of voxels to be included in calculation
        data_all (numpy): numpy array of pairwise stimulus data
        stimuli_indices_all (list): list of indices for all pairwise similarities
        stimuli_indices_compute (list): list of indices for which pairs to compute pairwise similarities

    Returns:
        list: pairwise similarities of sentence stimuli
    """
    
    # Get list of sentence pairs
    sent_id_pairs = list(itertools.combinations(stimuli_indices_all, 2))

    # Compute and store pairwise cosine similarities
    neuro_sim_storage = np.array([])
    for sent_id_pair in sent_id_pairs:
        if sent_id_pair[0] in stimuli_indices_compute and sent_id_pair[1] in stimuli_indices_compute:
            pair_sim = cosine_sim(data_all[sent_id_pair[0],voxel_indices_for_calc],data_all[sent_id_pair[1],voxel_indices_for_calc])
            neuro_sim_storage = np.append(neuro_sim_storage,pair_sim)
        else:
            neuro_sim_storage = np.append(neuro_sim_storage,np.NaN)
    
    return neuro_sim_storage


# Function to compute pairwise stimulus similarities across voxels
def compute_neuro_pairwise_sims_simple(data_all, sent_id_pairs):
    """ Compute the cosine similarity between all sentences over specified voxels

    Args:
        data_all (numpy): numpy array of pairwise stimulus data
        send_id_pairs (list): time indexes corresponding to occurences of sentence stimuli

    Returns:
        list: pairwise similarities of sentence stimuli
    """
    
    # Compute and store pairwise cosine similarities
    neuro_sim_storage = np.array([])
    for sent_id_pair in sent_id_pairs:
        pair_sim = cosine_sim(data_all[sent_id_pair[0],:], data_all[sent_id_pair[1],:])
        neuro_sim_storage = np.append(neuro_sim_storage,pair_sim)
    
    return neuro_sim_storage



# Get list of pairs of sentence ids that are in the test set, in the right order
def get_sentence_pairs_ids_test_set(stimuli_indices_all, all_pair_combinations, test_stimuli):
    """ Computes a list of pairs of sentence ids which include sentences in the test set

    Args:
        stimuli_indices_all (list): list of all sentence indices
        all_pair_combinations (list): list of tuples of sentence pairs
        test_stimuli (list): indices of test stimuli

    Returns:
        list: pairs of sentence IDs
    """
    
    # dict to move from TR-indices (e.g. 372) to stimulus-indices (e.g. 147)
    times_stimuli_reindex_dict = dict(zip(stimuli_indices_all, np.arange(stimuli_indices_all.shape[0])))
    
    # perform reindexing
    test_stimuli_reindexed = [times_stimuli_reindex_dict[index] for index in test_stimuli]
    
    # get list of sentence pairs in correct order
    sent_id_pairs_test = list(itertools.combinations(test_stimuli_reindexed, 2))
    sent_id_pairs_test_sorted = []
    for pair in sent_id_pairs_test:
        pair = list(pair)
        pair.sort()
        pair = tuple(pair)
        sent_id_pairs_test_sorted.append(pair)
    sent_id_pairs_test_sorted.sort()
    
    # Calculate indices for each pair of sentence IDs in the test set
    pairs_id_test = []
    for pair_indx,pair in enumerate(all_pair_combinations):
        if pair in sent_id_pairs_test: # only pairs in the test set
            pairs_id_test.append(pair_indx) # store for computing correlation
            
    return pairs_id_test


# Calculate correlations between model similarities and neuro similarities
def calc_neuro_model_correlations(neuro_sims, model_sims, control_vars, models_to_plot, subset, method, shuffle=False):
    """ Function to calculate the correlation between sentence similarities of neuro data and sentence similarites each model.

    Args:
        neuro_sims (dict): storage dictionary containing set of neuro similarities
        model_sims (dict): storage dictionary of sentence similarities for each model
        control_vars (list): list of zero or more upper triangle lists variables to use as controls for partial correlation computation
        models_to_plot (list): list of models for which to compute correlations
        subset (array): method used for selecting voxels over which to compute neuro similarity
        method (str): specify to use spearman or pearson correlation 
        shuffle (bool, optional): option to shuffle order of neuro similarities for control case, defaults to False

    Returns:
        dict: storage dictionary containing correlations and p-values for each model
    """
    raw_correlations = {}
    partial_correlations = {}
    
    # prepare variables for regression
    neuro_sims_subset = neuro_sims[subset]
    control_vars_subset = [control_var[subset] for control_var in control_vars]
    
    # loop over all models
    for model_name in models_to_plot:
        
        # shuffle elements of array for a control condition
        if shuffle==True:
            neuro_sims_subset = np.random.permutation(neuro_sims_subset)
            
        # prepare model sims for regression
        model_sims_subset = model_sims[model_name][subset]

        # compute raw correlations
        try:
            if method=='spearman':
                raw_correlations[model_name] = spearmanr(neuro_sims_subset, model_sims_subset)[0]
            elif method=='pearson':
                raw_correlations[model_name] = pearsonr(neuro_sims_subset, model_sims_subset)[0]
        except ValueError:
            print('Only',len(subset),'values')
                             
        # prepare data for computation of partial correlations
        all_data = np.array([neuro_sims_subset, model_sims_subset])
        col_names = ['neuro', 'model']
        control_var_names = []
        for idx,control_var_subset in enumerate(control_vars_subset): # store control variables
            all_data = np.vstack((all_data, control_var_subset))
            control_var_names.append('control_'+str(idx))
            col_names.append('control_'+str(idx))
        df = pd.DataFrame(np.array(all_data), index=col_names).T
        
        # compute partial correlations
        try:
            partial_correlations[model_name] = partial_corr(data=df, x='model', y='neuro', covar=control_var_names, method=method).values[0][0:2].tolist()[1]
            # partial_correlations[model_name] = -np.linalg.inv(np.corrcoef(all_data))[0,1]
        except ValueError:
            print('Only',len(subset),'values')
    
    return raw_correlations, partial_correlations

# Regress out a set of control variables from a dictionary of similarity scores
def regress_out_controls(y_vals_dict, control_vars):
    """ Takes a dictionary of similarity scores and returns a new dictionary with control variables regressed out

    Args:
        y_vals_dict (dict): dictionary of similarities
        control_vars (list): list of control variables to regress out

    Returns:
        dict, dict: dictionary of similarities with controls regressed out, and dictionary of partial correlations
    """
    # Define output dictionary
    y_vals_out = {}
    pcors_out = {}
    
    for key in y_vals_dict.keys():
        # Prepare data for regression
        y_vals = y_vals_dict[key]
        y_vals = np.nan_to_num(y_vals) # replace NaNs with zeros
        X = np.array(control_vars).T
        y = y_vals.reshape(-1,1)

        # Conduct regression
        OLS_reg = LinearRegression().fit(X, y)
        y_pred = OLS_reg.predict(X)
        resids = (y-y_pred).reshape(-1)
        
        # Get partial correlation
        partial_corr = np.sqrt(OLS_reg.score(X, y))
        
        # Save results
        y_vals_out[key] = resids
        pcors_out[key] = partial_corr
        
    return y_vals_out, pcors_out


# Add subplot for full stimuli scatterplot figure
def add_scatterplot_subplot(fig, xs, ys, colours, plot_num, neur_sim_type, model_name, x_min, y_min, y_max, size):
    """ Adds a scatterplot subplot to compare model similarity and neuro similarity data

    Args:
        fig (matplotlib_ax): axis object for figure
        xs (np_array): model similarity data
        ys (np_array): neuro similarity data
        colours (np_array): dataset to use for colouring sims
        plot_num (int): number of subplot
        neur_sim_type (str): method of voxel selection used
        model_name (str): model to be plotted
        x_min (int): lowest x value to include in regression and plot
        y_min (int): lowest y value to include in regression and plot
        y_max (int): highest y value to include in regression and plot
        size (int): size of dots
    """
    
    # create figure subplot   
    fig_axis = fig.add_subplot(*plot_num) 
    
    # plot data, errors, and reg data
    fig_axis.scatter(xs, ys, c=colours, s=size, edgecolor='none') # add data
    included_values = np.argwhere(xs > x_min).flatten() # remove any values less than x_min
    xs_poly = PolynomialFeatures(degree=1, include_bias=True).fit_transform(xs.reshape(-1, 1))
    poly_regression = LinearRegression()
    poly_regression.fit(xs_poly[included_values], ys[included_values])
    
    # plot regression line
    x_range = np.arange(x_min+0.01,0.99, step=0.01)
    x_range_poly = PolynomialFeatures(degree=1, include_bias=True).fit_transform(x_range.reshape(-1, 1))
    pred_ys = poly_regression.predict(x_range_poly)
    fig_axis.plot(x_range, pred_ys, color='red', linewidth=0.7) # add reg line
    corr = spearmanr(ys[included_values], xs[included_values])
    fig_axis.text(x_min+0.03, y_max-0.15, '{}, {}: {:.3f}'.format(model_name, neur_sim_type, corr[0])) # add labels with correl
    
    # add labels and axis ticks
    fig_axis.set(ylabel='neuro sims', xlabel='model sims')
    fig_axis.set_xlim(x_min,1)
    fig_axis.set_ylim(y_min,y_max)
    # fig_axis.set_title(model_name)
    # fig_axis.tick_params(axis='both', which='major', pad=-4)
    fig_axis.tick_params(axis='both', which='minor', pad=-4)
    # fig_axis.set_xticks(np.arange(min_threshold,1,0.2)) # use labels="" to turn off ticks
    # fig_axis.set_yticks(np.arange(-0.75,1,0.25))


# Samples observations in equal numbers in steps along the x-axis
def resample_xs_and_ys(xs, ys, sample_n, min_x, replace_draws, bin_step):
    """ This function is used to resample from a set of x and y values so that there is a more even
        distribution of values along the x axis.

    Args:
        xs (ndarray): x values to resample
        ys (ndarray): y values to resample
        sample_n (int): number of observations to draw within each bin
        replace_draws (bool): specifies whether to replace draws or not
        bin_step (int): width of bin interval, usually around 0.1

    Returns:
        (ndarray,ndarray): resampled x and y values
    """
    
    # Storage lists
    sampled_x_values = []
    sampled_y_values = []

    # Loop through the increments of bin_step and sample 50 data points for each increment
    for increment in np.arange(min_x, 1.1, bin_step):
        # Find the indices of x values that fall within the specified increment
        indices = np.where((xs >= increment) & (xs < increment + bin_step))[0]

        if len(indices) >= 5:
            # Sample 50 indices from the filtered indices
            sampled_indices = np.random.choice(indices, sample_n, replace=replace_draws)
            sampled_x_values.extend(xs[sampled_indices])
            sampled_y_values.extend(ys[sampled_indices])
            
    return np.array(sampled_x_values),np.array(sampled_y_values)


# Compute pooled standard deviation
def pooled_stdev(std_devs, ns):
    """ Simple function to compute pooled standard deviation

    Args:
        std_devs (list): standard deviations to pool
        ns (list): size of each set

    Returns:
        float: pooled standard deviation
    """
    vars = []
    n_mins_ones = []
    for std_dev in std_devs:
        vars.append(std_dev**2)
    for n in ns:
        n_mins_ones.append(n-1)
    numerator = np.dot(vars,n_mins_ones)
    denominator = np.sum(n_mins_ones)
    
    return np.sqrt(numerator/denominator)


# From https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
def barplot_annotate_brackets(num1, num2, data, center, height, colour='black', yerr=None, dh=.05, barh=0, fs=None, maxasterix=3, ymax=0.2, multi_test=1, width=1):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    :ymax: largest y value to plot
    :multi_test: whether to perform bon feroni correction for multiple tests
    """
    asterix_factors = [5,10,10,10]
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.01
        # *** is p < 0.001
        # etc.
        text = ''
        p_cutoff = 0.05/multi_test # correction for multiple tests
        i=0
        while data < p_cutoff:
            text += '*'
            p_cutoff /= asterix_factors[i]
            i+=1
            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = '‾'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(0,max(ly, ry)) + dh

    barx = [lx+0.07, lx+0.07, rx-0.07, rx-0.07]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh-0.03*ymax)

    plt.plot(barx, bary, c=colour, linewidth=width)

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
    

# Function to produce and save a .nii file from a 1d data array, along with a transform matrix and original image.nii file
def generate_nifti(data_1d, transform_1d_to_3d, image_nii, output_path):
    """ Save a 1D array to a .nii image

    Args:
        data_1d (np_array): 1D numpy array to convert to .nii
        transform_1d_to_3d (np_array): transformation matrix to convert 1D to 3D array
        image_nii (nii_image): nii image to get affine transformation for
        output_path (str): file path to save .nii file to
    """

    # Create empty 3d array
    max_coords = np.max(transform_1d_to_3d, axis=0)
    shape = tuple(max_coords + 1)
    data_3d = np.nan(shape)
    print(shape)

    # Save 1d data into 3d array
    for voxel_id,voxel_value in enumerate(data_1d):
        coords = transform_1d_to_3d[voxel_id]
        data_3d[coords[0],coords[1],coords[2]] = voxel_value

    # get transformation from structural
    affine_path = nibabel.load(image_nii).affine
    
    # Save numpy array as nifti
    print(data_3d.shape)
    ni_img = nibabel.Nifti1Image(data_3d, affine_path)
    nibabel.save(ni_img, output_path)
    
    

# Function to select the control variables we want to use for each plot
def get_control_vars(controls_type, model_sims_storage):
    """ Returns a list of control variables to use for partial correlation computation
    
    Args:
        controls_type (int): type of control variables to use
        model_sims_storage (dict): storage dictionary containing model similarities
    """
    sent_length_diffs_vector = model_sims_storage['diff_length']
    sent_length_mins_vector = model_sims_storage['min_length']
    visual_sims_vector = model_sims_storage['visual_sims']

    if controls_type==0:
        control_vars = [np.ones(len(sent_length_diffs_vector))] # no controls
    elif controls_type==1:
        control_vars = [sent_length_diffs_vector]
    elif controls_type==2:
        control_vars = [sent_length_diffs_vector,sent_length_mins_vector]
    elif controls_type==3:
        control_vars = [visual_sims_vector] # visual cortex similarities
    elif controls_type==4:
        control_vars = [sent_length_mins_vector]
    return control_vars


# Function to load models to include in the analysis
def get_included_models(model_sims_storage, models_shortnames):
    """ Returns a list of models to include in the analysis based on partial text matches
    
    Args:
        model_sims_storage (dict): storage dictionary containing model similarities
        models_shortnames (list): list of shortnames of models to include
    """
    included_models = {} # use dict to avoid duplicates
    for shortname in models_shortnames:
        for modelname in model_sims_storage.keys():
            if shortname in modelname: # get partial text matches
                included_models[modelname] = 1 # placeholder value
    included_models = list(included_models.keys())
    print('Included models:',included_models)
    
    return included_models


# Function for computing partial correlation
def simple_partial_corr(expr_sims, model_sims, ctrl_sims, method):
    """ Computes the partial correlation between two sets of similarity scores, controlling for a set of controls
    
    Args:
        expr_sims (np_array): experimental fmri similarities
        model_sims (np_array): model similarities
        ctrl_sims (list): list of control variable similarities
        method (str): specify to use spearman or pearson correlation
    """
    # prepare data for computation of partial correlations
    all_data = np.array([expr_sims, model_sims])
    col_names = ['expr', 'model']
    control_var_names = []
    for idx,control_var in enumerate(ctrl_sims): # store control variables
        all_data = np.vstack((all_data, control_var))
        control_var_names.append('control_'+str(idx))
        col_names.append('control_'+str(idx))
    df = pd.DataFrame(np.array(all_data), index=col_names).T

    # compute partial correlations
    result = partial_corr(data=df, x='expr', y='model', covar=control_var_names, method=method).values[0][0:2].tolist()
    
    return result

# function to convert a flattened upper triangle to a full nxn matrix
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


# Loads a set of sentence similarities, depending on whether it is a paired or non-paired list
def load_set_of_sentences(dataset_path: str, pairs: bool):
    """ Loads a full set of sentence similarities

    Args:
        dataset_path (str): path to dataset to load
        pairs (bool): load sentence pairs or a list of sentences

    Returns:
        dict: dictionary of embeddings
    """
    
    # Open dataset file
    sentences_dict = {}
    with open(dataset_path, encoding='utf-8') as file:
        lines = file.readlines()

    # Load data into dictionary
    if pairs==True: # for pairs of sentences
        # Save all sentence pairs into dictionary
        sentences_dict = {}
        i=1
        for item in lines: 
            item = item.strip()
            item_list = item.split(';')
            similarity = float(item_list[2])
            sent_pair = (item_list[0],item_list[1])
            sentences_dict[i] = [item_list[0],item_list[1],similarity]
            i=i+1
    elif pairs==False: # for single sentences
        sentences_dict = {}
        i=1
        for item in lines: 
            item = item.strip()
            sentences_dict[i] = item
            i=i+1
    
    return sentences_dict