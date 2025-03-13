"""
For balancing sample so that all blocks will have 18 tasks.
"""

import pandas as pd
import os
import numpy as np
import argparse

from python.utils.utils import map_nans

parser = argparse.ArgumentParser()

parser.add_argument("--sample_filepath", help="Directory name where sample is retrieved from.",
						type=str, required=True)

def main():

    args = parser.parse_args() # Parse arguments

    SAMPLE_FILEPATH = os.path.join("output", args.sample_filepath) # Retrieve sample file

    df_sample = pd.read_csv(SAMPLE_FILEPATH, na_filter=True)
    
    #################
    ## FILTER DATA ##
    #################

    map_nans(df_sample)

    df_sample = df_sample.dropna(subset = ["tgt_id"], axis='rows').reset_index(drop=True) # Drop images where only one target, 498 samples 
    idx = df_sample.index[(df_sample.absent == True) & (df_sample.absent_tgt_id.isna())] # Drop images where absent target is corruptedly sampled --> missing, 20 samples
    df_sample = df_sample.drop(idx).reset_index(drop=True)
    
    #####################
    ## END FILTER DATA ##
    #####################

    block_ids = np.unique(df_sample.block_id)
    balanced_block_ids = np.repeat(block_ids, 18)
    df_sample["balanced_block_id"] = balanced_block_ids[0:len(df_sample)]
                                                        
    print(f"Number of trials: {len(df_sample)}")
    df_sample.to_csv(os.path.join("output", "sample", "vsgui10k_sample_balanced.csv"))

if __name__ == "__main__":
    main()