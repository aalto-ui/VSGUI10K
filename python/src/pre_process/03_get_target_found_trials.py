import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from python.utils.utils import get_nearest_point

def main():
    """
    Estimates distance between fixation and the target (bounding box, distance is zero inside the target).

    Saves data to output/data/tgt_found_trials.csv
    """
    
    # Get fixation info
    df_fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))

    # Rename for clarity
    validation = df_fixations

    # Get target centers with respect to the display (not the screenshot), starting with the absent square
    width, height = 1920, 1200
    absent_x_center,  absent_y_center = (width - 200 + 75) / width, (height - 100 + 25) / height # Coordinates of the red square
    absent_x, absent_y = (width - 200) / width, (height-100) / height # Coordinates of the red square

    # Iterate over targets
    validation["tgt_x_center"] = (1-validation.scale_x) / 2 + validation.scale_x * validation.tgt_x + validation.scale_x * validation.tgt_width / 2 # Left pad + scaled target left edge + half of scaled target width
    validation["tgt_y_center"] = (1-validation.scale_y) / 2 + validation.scale_y * validation.tgt_y + validation.scale_y * validation.tgt_height / 2
    validation["tgt_x_scaled"] = (1-validation.scale_x) / 2 + validation.scale_x * validation.tgt_x # Just the upper-left corner
    validation["tgt_y_scaled"] = (1-validation.scale_y) / 2 + validation.scale_y * validation.tgt_y
    validation["tgt_height_scaled"] = validation.scale_y * validation.tgt_height # Scaled height
    validation["tgt_width_scaled"] = validation.scale_x * validation.tgt_width # Scaled width

    # Store info for absent trials
    validation.loc[validation["absent"] == True, ["tgt_x_center"]] = absent_x_center
    validation.loc[validation["absent"] == True, ["tgt_y_center"]] = absent_y_center
    validation.loc[validation["absent"] == True, ["tgt_x_scaled"]] = absent_x
    validation.loc[validation["absent"] == True, ["tgt_y_scaled"]] = absent_y
    validation.loc[validation["absent"] == True, ["tgt_height_scaled"]] = 150 / width
    validation.loc[validation["absent"] == True, ["tgt_width_scaled"]] = 50 / height

    # Find nearest point on target bounding box perimeter
    nearest_x_on_tgt_perimeter = []
    nearest_y_on_tgt_perimeter = []
    for row in tqdm(validation.itertuples()):
        nearest_x, nearest_y = get_nearest_point(row.tgt_x_scaled, row.tgt_y_scaled, row.tgt_width_scaled, row.tgt_height_scaled, row.FPOGX_debias, row.FPOGY_debias)
        nearest_x_on_tgt_perimeter.append(nearest_x)
        nearest_y_on_tgt_perimeter.append(nearest_y)

    validation["nearest_x_on_tgt_perimeter"] = nearest_x_on_tgt_perimeter
    validation["nearest_y_on_tgt_perimeter"] = nearest_y_on_tgt_perimeter

    # Set box for target absent trials
    validation.loc[validation["absent"] == True, ["tgt_x_center"]] = absent_x_center
    validation.loc[validation["absent"] == True, ["tgt_y_center"]] = absent_y_center

    # Calculate horizontal and vertical distance
    validation["tgt_distance_x_perimeter"] = np.abs(validation.nearest_x_on_tgt_perimeter - validation.FPOGX)
    validation["tgt_distance_y_perimeter"] = np.abs(validation.nearest_y_on_tgt_perimeter - validation.FPOGY)
    
    # Calculate Euclidian distance between the target center and fixation in pixels and normalised. Set as zero if inside the target.
    validation["tgt_distance_x_perimeter"] = (validation["tgt_distance_x_perimeter"]).where(~((validation["FPOGX_debias"] > validation["tgt_x_scaled"]) &
     (validation["FPOGX_debias"] < validation["tgt_x_scaled"] + validation["tgt_width_scaled"])), other = 0)
    validation["tgt_distance_y_perimeter"] = (validation["tgt_distance_y_perimeter"]).where(~(
     (validation["FPOGY_debias"] > validation["tgt_y_scaled"]) &
     (validation["FPOGY_debias"] < validation["tgt_y_scaled"] + validation["tgt_height_scaled"])), other = 0)
    
    # Store in pixels and normalized
    validation["tgt_distance_px"] = np.sqrt((validation.tgt_x_center*width-validation.FPOGX*width)**2 + (validation.tgt_y_center*height-validation.FPOGY*height)**2)
    validation["tgt_distance"] = np.sqrt((validation.nearest_x_on_tgt_perimeter-validation.FPOGX)**2 + (validation.nearest_y_on_tgt_perimeter-validation.FPOGY)**2)

    validation["tgt_distance_perimeter"] = np.sqrt((validation.nearest_x_on_tgt_perimeter-validation.FPOGX)**2 + (validation.nearest_y_on_tgt_perimeter-validation.FPOGY)**2)

    
    validation["tgt_distance_perimeter"] = (validation["tgt_distance_perimeter"]).where(
                                                                                        ~((validation["FPOGX_debias"] > validation["tgt_x_scaled"]) &
                                                                                          (validation["FPOGX_debias"] < validation["tgt_x_scaled"] + validation["tgt_width_scaled"]) &
                                                                                          (validation["FPOGY_debias"] > validation["tgt_y_scaled"]) &
                                                                                           (validation["FPOGY_debias"] < validation["tgt_y_scaled"] + validation["tgt_height_scaled"]) ), other = 0)
    
    validation["tgt_distance_perimeter_px"] = np.sqrt((validation.nearest_x_on_tgt_perimeter*width-validation.FPOGX*width)**2 + (validation.nearest_y_on_tgt_perimeter*height-validation.FPOGY*height)**2)

        
    validation["tgt_distance_perimeter_px"] = (validation["tgt_distance_perimeter_px"]).where(
                                                                                        ~((validation["FPOGX_debias"] > validation["tgt_x_scaled"]) &
                                                                                          (validation["FPOGX_debias"] < validation["tgt_x_scaled"] + validation["tgt_width_scaled"]) &
                                                                                          (validation["FPOGY_debias"] > validation["tgt_y_scaled"]) &
                                                                                           (validation["FPOGY_debias"] < validation["tgt_y_scaled"] + validation["tgt_height_scaled"]) ), other = 0)
    # Save target found data
    os.makedirs(os.path.join("output", "data"), exist_ok=True)
    validation.to_csv(os.path.join("output", "data", "tgt_found_trials.csv"), index=False)
    
if __name__=="__main__":
    main()