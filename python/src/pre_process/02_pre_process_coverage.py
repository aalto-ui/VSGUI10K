import pandas as pd
import numpy as np
from tqdm import tqdm
import shapely
import os
from PIL import Image

from python.utils.utils import get_coverage, get_coverage_mixed


def main():
    """
    Pre-processes fixations to estimates of foveal coverage based on viewing distance. Used in Fig. 5.

    Saves search time file to output/data/coverage_{viewing_distance}.csv
    """

    # CHANGE VIEWING DISTANCE HERE
    distance = 50

    # Read fixation data from CSV file
    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))

    # Initialize dictionary to store coverage information
    coverage_dict = {k: [] for k in ["coverage_img", "pid", "coverage_all", "category", "absent", "cue", "n", "max_n", "img_name"]}

    # Iterate over categories (web, desktop, mobile)
    for category in ["web", "desktop", "mobile"]:
        images = np.unique(fixations[fixations.category == category].img_name)
        
        # Iterate over images within the category
        for img_name in tqdm(images):

            # Filter fixations for the current image
            fixations_sample = fixations[fixations.img_name == img_name]

            # Get unique participant IDs for the current image
            pid_subset = np.unique(fixations_sample.pid)

            # Get scaling factors for the current image (vertical and horizontal span of the image on the experiment screen)
            scale_x = np.unique(fixations_sample.scale_x)[0]
            scale_y = np.unique(fixations_sample.scale_y)[0]
            
            for pid in pid_subset:
                # Filter fixations for the current participant
                sample_pid = fixations_sample[fixations_sample.pid == pid]
                
                # Extract image name, absent indicator, and cue from the sample
                img_name = np.array(sample_pid.img_name)[0]
                absent = np.array(sample_pid.absent)[0]
                cue = np.array(sample_pid.cue)[0]

                # Calculate the radius in pixels for a degree of visual angle, assume PPI 94
                radius = np.tan(np.radians(1)) * (distance / 2.54) * 94

                # Set base dimensions for the screen, based on the monitor used in the study
                base_height = 1200
                base_width = 1980

                # Calculate the position and size of the image on the screen
                _x = base_width * (1 - scale_x) / 2 # Upper-left corner
                _y = base_height * (1 - scale_y) / 2 # Upper-left corner
                _height = scale_y * base_height
                _width = scale_x * base_width
                
                # Create a rectangle polygon for the image boundaries
                coords = ((_x, _y), (_x, _y + _height), (_x + _width, _y + _height), (_x + _width, _y), (_x, _y))
                rectangle = shapely.Polygon(coords)

                # Iterate over fixations
                for n in range(len(sample_pid)):
                    sample_until = sample_pid[0:n]

                    # Calculate coverage areas
                    area, _ = get_coverage(sample=sample_until, radius=radius, original_height=1200, original_width=1920, col_x="FPOGX_debias", col_y="FPOGY_debias")
                    area_mixed = get_coverage_mixed([rectangle, area])

                    # Store coverage metrics in the dictionary
                    coverage_dict["coverage_img"].append(area_mixed.area / (_height * _width))
                    coverage_dict["coverage_all"].append(area.area / (1920 * 1200))
                    coverage_dict["pid"].append(pid)
                    coverage_dict["category"].append(category)
                    coverage_dict["absent"].append(absent)
                    coverage_dict["cue"].append(cue)
                    coverage_dict["n"].append(n)
                    coverage_dict["max_n"].append(len(sample_pid))
                    coverage_dict["img_name"].append(img_name)

    # Convert coverage dictionary to DataFrame
    coverage_df = pd.DataFrame(coverage_dict)

    # Save coverage DataFrame to CSV file
    coverage_df.to_csv(os.path.join("output", "data", f"coverage_{distance}.csv"), index=False)

if __name__=="__main__":
    main()