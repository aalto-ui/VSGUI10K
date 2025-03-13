import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from CONFIG import UEYES_DIR # UEyes data stored outside this repo, make sure to configure path in untracked_config/untracked_config.py

def main():
    """
    Pre-processes UEyes data from per participant files to one data file. Required to plot heatmaps, e.g. Fig. 10. Also remember to run python.src.pre_process.ueyes.01_get_dims.

    Saves fixation file to output/data/ueyes_fixations.csv
    """

    # Retrieve UEyes data
    data_dir = os.path.join(UEYES_DIR, "eyetracker_logs")

    fixation_files = [name for name in os.listdir(os.path.join(data_dir)) 
                                if "fixations" in name]
    
    # Get sample file for matching info with UEyes data
    df_sample = pd.read_csv(os.path.join("output", "sample", "vsgui10k_sample_balanced.csv")).reset_index()
    ueyes_img_names = np.unique(df_sample.img_name)

    # Concatenate all
    all_dfs = []
    for fixation_file in tqdm(fixation_files):
        try:
            df_tmp = pd.read_csv(os.path.join(data_dir, fixation_file))
            df_tmp = df_tmp[df_tmp.MEDIA_NAME.isin(ueyes_img_names)]#.sort_index()
            all_dfs.append(df_tmp)
        except Exception as e:
            print (f"Issue with file {fixation_file} : {e}")
            continue

    # Choose columns
    all_data = pd.concat(all_dfs)[["MEDIA_NAME", "FPOGX", "FPOGY", "FPOGS", "FPOGD","FPOGID","FPOGV","BPOGX","BPOGY","BPOGV"]]

    print ("Concat done")

    # Merge UEyes with sample data
    all_data = all_data.merge(df_sample[["category", "img_name"]].drop_duplicates(), how="left", right_on="img_name", left_on="MEDIA_NAME").reset_index(drop=True).drop_duplicates()

    print ("Categories merged")

    # Save UEyes fixation data
    all_data[["FPOGX", "FPOGY", "FPOGS", "FPOGD","FPOGID","FPOGV","BPOGX","BPOGY","BPOGV", "category", "img_name"]].to_csv(os.path.join("output", "data", "ueyes_fixations.csv"), index=False)

if __name__=="__main__":
    main()
