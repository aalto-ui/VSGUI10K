import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    """
    Plot Fig 9 from AIM metrics (visual clutter and color variability).
    Saves plots to output/figs/supplement
    """

    # Configure directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Retrieve data
    results = pd.read_csv(os.path.join("output", "data", "vsgui10k_aim_results_with_factors.csv"))

    # Columns pertaining to visual clutter and color variability metrics, refer to AIM
    clcv_cols = ['m1_0', 'm2_0', 'm3_0', 'm4_0', 'm7_0', 'm8_0', 'm11_0', 'm12_0', 'm19_0', 'Factor1', 'Factor2', 'Factor3']

    # Labels of the columns
    clcv_labels = ['PNG file size\n(B, CV1)', 'JPEG file size\n(B, CL4)', 'Distinct RGB values\n(#, CV2)', 'Contour density\n(proportion, CL1)', 'Subband entropy\n(CL2)', 'Feature congestion\n(proportion, CL3)', 'Static clusters\n(#, CV4)', 'Dynamic clusters\n(#, CV5)', 'Distinct RGB values\nper dynamic cluster\n(#, CV3)']

    # Paleltte to be used for grouping
    clcv_palette = ["#D81B60", "#1E88E5", "#FFC107", "lightgrey"] #"#004D40"]

    # Define which color belongs to which metric
    clcv_colors = [2,1,2,3,1,0,0,0,3,0,1,2]

    # Create dictionaries of the metrics
    clcv_dict = {k : i for k,i in zip(clcv_cols, clcv_labels)}

    # Rename factor scores
    clcv_dict["Factor1"] = "Factor 1"
    clcv_dict["Factor2"] = "Factor 2"
    clcv_dict["Factor3"] = "Factor 3"

    # Define order for CLCV metrics
    clcv_order = ['m1_0', 'm3_0', 'm19_0', 'm11_0', 'm12_0', 'm4_0', 'm7_0', 'm8_0', 'm2_0', 'Factor1', 'Factor2', 'Factor3']

    fig, axs = plt.subplots(4,3, figsize = (6, 8))

    row = 0
    col = 0

    # Plot
    for index, metric in enumerate(clcv_order):
        ax = axs[row,col]

        sns.histplot(data=results, x=metric, ax=ax, bins=20, color=clcv_palette[clcv_colors[index]], stat="percent")
        ax.set_xlabel(clcv_dict[metric])

        print (f"{clcv_dict[metric]}: [{np.around(results[metric].min(), 2)}, {np.around(results[metric].max(), 2)}]")
        ax.set_ylim(0,40)

        if col > 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
        else:
            ax.set_ylabel("Percentage")

        col += 1
        if col > 2:
            row+=1
            col = 0

    plt.subplots_adjust(wspace=0.1, hspace=0.8)

    plt.savefig(os.path.join(save_dir, f"supp_fig9.pdf"),dpi=300,bbox_inches="tight")


if __name__ == "__main__":
    main()