import pandas as pd
import os
import matplotlib as mpl

from python.utils.utils import plot_three_step

def main():
    """
    Plot Fig. 11 for Guess-Scan-Confirm. Note that the function plot_three_step can be used to inspect different Guess-Scan-Confirm cutoffs.

    Also plot Figs. 23-24 in the Supplementary Materials.

    Saves plots to output/figs
    """
    # Set for plotting
    mpl.rc('font', family='Arial')

    # Read fixation data from CSV file
    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))
    
    # Filter the fixation data to include only entries with visual search 
    fixations = fixations[fixations.img_type == 2]

    # Read target data from CSV file (required to plot target rectangle)
    df_target = pd.read_csv(os.path.join("data", "vsgui10k_targets.csv"))

    # Compute the maximum fixation identifier (FPOGID) for each image for filtering
    grouped = fixations.groupby(["new_img_name"]).agg({"FPOGID": "max", "absent": "mean", "category": "count"}).reset_index()
    grouped = grouped.rename({"FPOGID": "FPOGID_MAX"}, axis="columns")
    fixations = fixations.merge(grouped[["new_img_name", "FPOGID_MAX"]], how="left", on="new_img_name")
    fixations["FPOGID_NORM"] = fixations["FPOGID"] / fixations["FPOGID_MAX"]

    # Call the plot_three_step function to plot the heatmaps
    plot_three_step(FPOGID_min=3, FPOGID_max_lower_bound=7, FPOGID_max_upper_bound=fixations.FPOGID_MAX.max(), 
                    scan_lower=5, scan_higher=2, filename="fig11.pdf", fixation_cross=False, 
                    fixations=fixations, df_target=df_target)
    
    plot_three_step(FPOGID_min = 0, FPOGID_max_lower_bound = 0, FPOGID_max_upper_bound = 13, scan_lower = 0.2, scan_higher = 0.8, filename="supplement/supp_fig23.pdf", fixation_cross=False, fixations=fixations, df_target=df_target) # Prints for supplement
    
    plot_three_step(FPOGID_min = 0, FPOGID_max_lower_bound = 13, FPOGID_max_upper_bound = fixations.FPOGID_MAX.max(), scan_lower = 0.2, scan_higher = 0.8, filename="supplement/supp_fig24.pdf", fixation_cross=False, fixations=fixations, df_target=df_target) # Prints for supplement

if __name__ == "__main__":
    main()
