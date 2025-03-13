# Generate sample so that we have n_samples 2 targets * 6 repeats * n images samples
# Produce CSV file with n_samples rows
# If the target is textual, sample from (text), (text, size), (text,color), (text,size,color)
# One participant should not see the same image twice, so each image occurence should be spaced out: one option to shuffle the images randomly this order repeat 6

from tqdm import tqdm
import argparse
import os
import pandas as pd
import numpy as np
from typing import Dict, List
import random

parser = argparse.ArgumentParser()

parser.add_argument("--target_filepath", help="Directory where annotated files are.",
						type=str, required=True)
parser.add_argument("--fixation_cross", help="Whether to use fixation cross.", action = "store_true")

def sample(img_names_sample : np.array, df_target : pd.DataFrame, n_targets : int, no_repeats_interval : int, n_repeats : int, tgt_dict : Dict, absent_dict : Dict, fixation_cross : bool, seed : int = 123) -> Dict:

    """
    Sample the experiment sequence.

    Attributes:
    -----------
    img_names_sample : np.array
        sample of images from UEyes (no duplicates)
    df_target : pd.DataFrame
        table containing info about all targets
    n_targets : int
        number of targets
    no_repeats_interval : int  
        how often the same image is allowed to repeat
    n_repeats : int
        number of repeats of each image / target pair
    tgt_dict : Dict
        dictionary containing image / target pairs
    absent_dict : Dict
        dictionary containing {image name : target that is absent (index in n_targets)} pairs
    seed : int
        random seed

    Returns:
    --------
    sample_dict : Dict
        dictionary containing the sequence for the experiment
    """

    # Seed
    np.random.seed(seed)
    random.seed(seed)

    # Repeat images in the sample
    img_names_sample = np.repeat(img_names_sample, n_repeats)
    
    # Initialise sample dictionary and counters
    sample_dict = {k : [] for k in ["img_name", "tgt_id", "absent", "text_description", "new_img_name", "block_id", "absent_img", "absent_tgt_id", "category"]}
    idx = 0
    block_id = 0
    block_counter = 0

    used_targets = []
    no_repeats_buffer = []

    # Sample
    for target in range(n_targets):

        # Collect used targets here
        absent_tgt_pairs = {}

        # Shuffle the images for each target
        sample, no_repeats_buffer = sample_names(img_names_sample, no_repeats_interval, no_repeats_buffer, seed=seed+target)

        for index, img_name in enumerate(tqdm(sample)):
            
            tgt_id = tgt_dict[img_name][target] # Set target 

            # If image is not in the absent_dict
            if img_name not in absent_dict.keys():

                absent_img = False
                absent_tgt_id = np.nan
                absent = False

            else: 
                # If image is in the absent_dict and this is the target index where target is absent, sample target from another image
                if absent_dict[img_name] == target:
                    absent_img_idx = index - target * len(sample) + no_repeats_interval 
                    absent = True
                    
                    if absent_img_idx >= len(sample):
                        absent_img_idx = index - len(sample) + no_repeats_interval - target * len(sample) 

                    absent_img = sample[absent_img_idx]

                    if absent_img not in absent_tgt_pairs.keys():
                        absent_tgt_sampled = False
                        df_tmp = df_target[df_target.img_name == absent_img]
                        absent_targets = np.unique(df_tmp.tgt_id)
                        
                        while absent_tgt_sampled != True: # Sample until a target not used is found
                            absent_tgt_id = np.random.choice(absent_targets, replace=False)
                            if absent_tgt_id not in used_targets:
                                used_targets.append(absent_tgt_id)
                                absent_tgt_sampled = True
                            elif absent_tgt_id in used_targets and len(absent_targets) <= 1:
                                absent_tgt_sampled = True
                                absent_tgt_id = np.nan

                        absent_tgt_pairs[absent_img] = absent_tgt_id

                    else:
                        absent_tgt_id = absent_tgt_pairs[absent_img]
                
                else:
                    absent_img = False
                    absent_tgt_id = np.nan
                    absent = False

            sample_dict["absent"].append(absent)
            #sample_dict["absent_img"].append(absent_img)
            if not absent_img:
                sample_dict["absent_img"].append(np.nan)
            else:
                sample_dict["absent_img"].append(np.unique(df_target[df_target.img_name == absent_img].ueyes_img_name)[0])
            
            sample_dict["absent_tgt_id"].append(absent_tgt_id)
            sample_dict["new_img_name"].append(f"{idx}.jpg")

            if fixation_cross: # Sample for fixation cross
                idx += 4
            else:
                idx += 3

            sample_dict["block_id"].append(f"{block_id}") # Record blocks
            block_counter += 1
            if block_counter >= 18:
                block_id += 1
                block_counter = 0

            sample_dict["category"].append(np.unique(df_target[df_target.img_name == img_name].category)[0])
            #sample_dict["img_name"].append(img_name)
            sample_dict["img_name"].append(np.unique(df_target[df_target.img_name == img_name].ueyes_img_name)[0])
            sample_dict["tgt_id"].append(tgt_id)
            sample_dict["text_description"].append(np.random.choice(["i", "t", "tc"], p=[0.5, 0.25, 0.25]))

    return sample_dict

def sample_names(img_names_sample : np.array, no_repeats_interval : int, no_repeats_buffer : List = [], seed : int = 123):

    """
    Shuffle the image names before each target.

    Attributes:
    -----------
    img_names_sample : np.array
        sample of images from UEyes (no duplicates)
    no_repeats_interval : int
        how often the same image is allowed to repeat
    no_repeats_buffer : List
        a list of n=no_repeats_interval last images, length is 0 if first time sampling
    seed : int
        random seed

    Returns:
    --------
    sample : np.array
        shuffled list of image names
    no_repeats_buffer : List
        what's left in the repeat buffer after shuffling
    """

    sample = []
    np.random.seed(seed)
    random.seed(seed)

    while len(img_names_sample) > 0:
        img_sampled = False
        while img_sampled == False:
            current_img_idx = np.random.choice(np.arange(len(img_names_sample)), size=1)
            current_img = img_names_sample[current_img_idx]# Select without replacement
    
            if current_img not in no_repeats_buffer:
                img_sampled = True
            
            if len(img_names_sample) < 30 and current_img not in no_repeats_buffer[200:]:
                 img_sampled = True

            if len(img_names_sample) < 3:
                img_sampled = True

        img_names_sample = np.delete(img_names_sample, current_img_idx)
        sample.append(current_img.tolist()[0])

        no_repeats_buffer.append(current_img)
        # The buffer is FIFO
        if len(no_repeats_buffer) > no_repeats_interval:
            no_repeats_buffer.pop(0)


    sample = np.array(sample)

    return sample, no_repeats_buffer

def sample_absent(img_names_sample : np.array, absent_prop : float = 0.2, n_targets : int = 2, seed : int = 123):

    """
    Sample images where one of the targets will be absent (at proportion 20%).

    Attributes:
    -----------
    img_names_sample : np. array
        sample of images from UEyes (no duplicates)
    absent_prop : float
        proportion of images that will have one target absent
    n_targets : int
        number of targets
    seed : int
        random seed

    Returns:
    --------
    absent_dict : Dict
        dictionary containing {image name : target that is absent (index in n_targets)} pairs
    """

    np.random.seed(seed)
    random.seed(seed)

    indices = np.arange(len(img_names_sample))
    absent_indices = np.random.choice(indices, size = int(absent_prop * len(indices))) # Sample indices where targets will be absent
    tgt_idx_for_absent = np.random.choice(np.arange(n_targets), size = len(absent_indices)) # Sample which target will be absent (first or second instance), resulting in around 10% absent targets in total

    absent_dict = {img_names_sample[absent_idx] : tgt_idx for absent_idx, tgt_idx in zip(absent_indices, tgt_idx_for_absent)}

    return absent_dict

def sample_targets(img_names_sample : np.array, df_target : pd.DataFrame, n_targets : int = 2, seed : int = 123):
    """
    Sample targets for each image.

    Attributes:
    -----------
    img_names_sample : np. array
        sample of images from UEyes (no duplicates)
    df_target : pd.DataFrame
        table containing info about all targets
    n_targets : int
        number of targets
    seed : int
        random seed

    Returns:
    --------
    tgt_dict : Dict
        dictionary containing image / target pairs
    """

    np.random.seed(seed)
    random.seed(seed)

    tgt_dict = {}

    for img_name in img_names_sample:
        df_tmp = df_target[df_target.img_name == img_name]
        if len(df_tmp) == 1:
            tgt_dict[img_name] = np.array([df_tmp.tgt_id.item(), np.nan]) # If only one target available, sample one and set one to NaN
        else:
            targets = np.unique(df_tmp.tgt_id)
            sample_targets = np.random.choice(targets, size=n_targets, replace=False) # If two targets available, sample both
            tgt_dict[img_name] = sample_targets

    return tgt_dict

def main():

    ####################
    ## Set parameters ##
    ####################

    n_repeats = 6 # How many repeats per image
    no_repeats_interval = 300 # At what interval can an image repeat
    n_images = 300 # Number of images per category
    n_targets = 2 # Number of targets per image
    seed = 112 # Seed

    ###################
    ## Get arguments ##
    ###################
    args = parser.parse_args()

    ######################
    ## Get target files ##
    ######################

    target_filepath = os.path.join("data", args.target_filepath)
    df_target = pd.read_csv(target_filepath)

    print (f"Number of rows in the df: {len(df_target)}; Number of targets: {len(np.unique(df_target.tgt_id))}")

    ###################
    ## Sample images ##
    ###################

    img_names_desktop = np.unique(df_target[df_target.category == "desktop"].img_name.dropna())
    img_names_mobile = np.unique(df_target[df_target.category == "mobile"].img_name.dropna())
    img_names_web = np.unique(df_target[df_target.category == "web"].img_name.dropna())

    np.random.seed(seed)
    random.seed(seed)
    img_names_desktop_sample = np.random.choice(img_names_desktop, size=n_images, replace=False)
    img_names_mobile_sample = np.random.choice(img_names_mobile, size=n_images, replace=False)
    img_names_web_sample = np.random.choice(img_names_web, size=n_images, replace=False)

    img_names_sample = np.concatenate([img_names_desktop_sample, img_names_mobile_sample, img_names_web_sample])
        
    ##############################
    ## Create experiment sample ##
    ##############################

    tgt_dict = sample_targets(img_names_sample, df_target, seed = seed) # Sample all targets

    absent_dict = sample_absent(img_names_sample, seed = seed) # Sample absent trials
    
    sample_dict = sample(img_names_sample = img_names_sample, df_target = df_target, tgt_dict=tgt_dict, absent_dict = absent_dict, n_targets=n_targets, no_repeats_interval=no_repeats_interval, n_repeats=n_repeats, fixation_cross=args.fixation_cross, seed=seed) # Consolidate

    ##################
    ## Store sample ##
    ##################

    df_sample = pd.DataFrame(sample_dict)

    os.makedirs(os.path.join("output", "sample"), exist_ok=True)
    df_sample.to_csv(os.path.join("output", "sample", "vsgui10k_sample.csv"))

if __name__ == "__main__":
    main()