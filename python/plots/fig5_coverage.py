# Note that you need to run <src.pre_process.02_pre_process_coverage> before this script

import os 
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import numpy as np
from PIL import Image
import shapely
import string
import shapely

from python.utils.utils import get_coverage, draw_scanpath

def main():
    """
    Plot Fig. 5 for demonstrating foveal coverage.
    """

    # Set for plotting
    mpl.rc('font', family='Arial')

    # Define a dictionary mapping categories to example image filenames
    image_dict = {"web": "ce9bfe.png", "desktop": "b889dd.png", "mobile": "0bdd56.jpg"}

    # Set distance to screen in cm
    distance = 50

    ##############
    ## Get data ##
    ##############

    # Read coverage data from CSV file
    try:
        coverage = pd.read_csv(os.path.join("output", "data", f"coverage_{distance}.csv"))
    except:
        print("Did you format coverage data? Run python.src.pre_process.02_pre_process_coverage script.")
        sys.exit()

    # Read fixation data from CSV file and filter for visual search tasks
    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))
    fixations = fixations[fixations.img_type == 2]

    # Read target data from CSV file
    df_target = pd.read_csv(os.path.join("data", "vsgui10k_targets.csv"))

    # Map values to descrptive strings
    coverage["absent"] = coverage["absent"].map({True: 'Target absent', False: 'Target present'})
    coverage["category"] = coverage["category"].map({"web": 'Webpage', "mobile": 'Mobile UI', "desktop": "Desktop UI"})
    
    ##########
    ## Plot ##
    ##########

    # Define grid layout specifications for subplots
    gs_kw = dict(width_ratios=[0.5, 1, 1], height_ratios=[1, 1, 1.5])

    # Create a figure using a mosaic layout
    axd = plt.figure(layout="constrained", figsize=(6, 5)).subplot_mosaic(
        """
        CBA
        FED
        GGG
        """, gridspec_kw=gs_kw)

    # Set titles and labels
    axd["A"].set_title("Webpage\nexample", size=12, weight="bold")
    axd["B"].set_title("Desktop UI\nexample", size=12, weight="bold")
    axd["C"].set_title("Mobile UI\nexample", size=12, weight="bold")

    axd["C"].set_ylabel("Target present", size=10)
    axd["F"].set_ylabel("Target absent", size=10)

    # Create a boxplot 
    sns.boxplot(
        ax=axd["G"], palette=["#1f449C", "#f05039"],
        data=coverage.groupby(["pid", "img_name", "category", "absent"]).agg({"coverage_img": "max"}).reset_index(),
        x="category", y="coverage_img", hue="absent", order=["Mobile UI", "Desktop UI", "Webpage"]
    )

    axd["G"].set_xlabel("")
    axd["G"].set_ylabel("Foveated area (proportion)", size=10)
    axd["G"].legend(title="", bbox_to_anchor=[0.75, -0.2], ncols=2)

    # Define some mappings and colors
    ax_mappings = {False: ["A", "B", "C"], True: ["D", "E", "F"]}
    colors = {False: "#1f449C", True: "#f05039"}

    # Seed for reproducible results
    np.random.seed(1234)

    # Iterate over presence/absence conditions and categories
    for absent in [False, True]:
        for index, category in zip(ax_mappings[absent], ["web", "desktop", "mobile"]):
            ax = axd[index]
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Filter fixations for the current category and condition
            fixations_sample = fixations[(fixations.img_name == image_dict[category]) & (fixations.absent == absent)]
            pid = np.random.choice(np.unique(fixations_sample.pid))

            # Filter fixations for the sampled participant
            fixations_sample = fixations_sample[fixations_sample.pid == pid]

            tgt_id = np.array(fixations_sample.tgt_id)[0]

            # Filter for the selected target
            df_target_sample = df_target[df_target.tgt_id == tgt_id]
            original_height, original_width = df_target_sample.original_height.item(), df_target_sample.original_width.item()

            # Calculate foveal coverage
            radius = np.tan(np.radians(1)) * (distance / 2.54) * 94
            area, n_points = get_coverage(
                sample=fixations_sample, radius=radius, original_height=original_height,
                original_width=original_width, col_x="FPOGX_scaled", col_y="FPOGY_scaled"
            )

            # Plot coverage on the screenshot
            try:
                for geom in area.geoms:
                    ax.plot(*geom.exterior.xy, color=colors[absent])
            except:
                ax.plot(*area.exterior.xy, color=colors[absent])

            ax.invert_yaxis() # Invert due to Gazepoint data orientation

            # Set limits
            ax.set_xlim(0, original_width)
            ax.set_ylim(original_height, 0)

            # Extract scanpath coordinates and durations. Note: use scaled data as plotted in screenshot dimensions.
            xs = fixations_sample.FPOGX_scaled
            ys = fixations_sample.FPOGY_scaled
            ts = fixations_sample.FPOGD

            # Get image
            img_path = os.path.join("data", "vsgui10k-images", image_dict[category])

            # Draw scanpath on the image
            _ = draw_scanpath(xs=np.array(xs), ys=np.array(ys), ts=np.array(ts), img_path=img_path, ax=ax, draw=False)
    
    # Save the figure to PDF file
    plt.savefig(os.path.join("output", "figs", f"fig5.pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

