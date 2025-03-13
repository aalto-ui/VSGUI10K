import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from CONFIG import UEYES_DIR # UEyes data stored outside this repo.

def main():
    """
    The UEyes experiment displayed images with a padding around them. This script maps the fixation coordinates to take into account the paddings.

    See: https://github.com/YueJiang-nj/UEyes-CHI2023/blob/7bc064175310284c74e64e0c3c2ef264bfbe66e2/data_processing/generate_heatmaps.py#L223

    Saves fixation file (in place) to output/data/ueyes_fixations.csv
    """
    # Get images to obtain dimensions
    img_dir = os.path.join(UEYES_DIR, "images")
    # Get sample file
    df_ueyes = pd.read_csv(os.path.join("output", "data", "ueyes_fixations.csv")).reset_index()

    img_names_ueyes = np.unique(df_ueyes.img_name)

    dims_dict = {k : [] for k in ["img_name", "original_width", "original_height"]}

    # Get dimensions (this) for each UEyes image
    for name in img_names_ueyes:
        img = Image.open(os.path.join(img_dir, name))
        dims_dict["img_name"].append(name)
        dims_dict["original_width"].append(img.size[0])
        dims_dict["original_height"].append(img.size[1])

    df_dims = pd.DataFrame(dims_dict)

    # Calculate relative locations of fixations
    screen_ratio = 1920 / 1200

    df_dims["img_ratio"] = df_dims.original_width / df_dims.original_height

    df_dims["size_x"] = df_dims.original_width.astype(float) # Change dtype to float to avoid error in the next step
    df_dims["size_y"] = df_dims.original_height.astype(float)

    df_dims.loc[screen_ratio >= df_dims.img_ratio, ["size_x"]] = df_dims.original_height / 1200 * 1920
    
    df_dims.loc[screen_ratio < df_dims.img_ratio, ["size_y"]] = df_dims.original_width / 1920 * 1200

    df_ueyes = df_ueyes.merge(df_dims, on="img_name")
        
    # get the image coordinates which is screen coords minus pad
    df_ueyes["FPOGX_scaled"] = (df_ueyes.FPOGX * df_ueyes.size_x - ((df_ueyes.size_x - df_ueyes.original_width) / 2)) / df_ueyes.original_width
    
    df_ueyes["FPOGY_scaled"] = (df_ueyes.FPOGY * df_ueyes.size_y - ((df_ueyes.size_y - df_ueyes.original_height) / 2)) / df_ueyes.original_height

    # Save
    df_ueyes.to_csv(os.path.join("output", "data", "ueyes_fixations.csv"), index=False)

if __name__=="__main__":
    main()

