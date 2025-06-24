import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    """
    Plot Fig 9 from AIM metrics (grid quality).
    Saves plots to output/figs/supplement
    """

    # Configure directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Retrieve data
    results = pd.read_csv(os.path.join("output", "data", "vsgui10k_aim_results_with_factors.csv"))

    # Columns pertaining to grid quality
    grid_cols = ['m21_0', 'm21_2', 'm21_4', 'm21_6', 'm21_8', "Factor4"]
    grid_labels = ["Number of visual\nGUI blocks (#, G1)", 'Number of alignment\npoints (#, G2)', 'Number of block\nsizes (#,G3)', "GUI coverage\n(proportion,G4)", "Number of vertical\nblock sizes (G5)"]

    grid_palette = ["#004D40", 'lightgrey']

    grid_colors = [0, 0, 0, 1, 0, 0]

    # Create dictionaries of the metrics
    grid_dict = {k : i for k,i in zip(grid_cols, grid_labels)}

    grid_dict["Factor4"] = "Factor 4"

    # Plot
    fig, axs = plt.subplots(2,3, figsize = (6, 4))

    row = 0
    col = 0

    for index, metric in enumerate(grid_cols):
        ax = axs[row,col]

        sns.histplot(data=results, x=metric, ax=ax, bins=20, color=grid_palette[grid_colors[index]], stat="percent")
        ax.set_xlabel(grid_dict[metric])

        print (f"{grid_dict[metric]}: [{np.around(results[metric].min(), 2)}, {np.around(results[metric].max(), 2)}]")


        ax.set_ylim(0,60)

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

    plt.savefig(os.path.join(save_dir, f"supp_fig10.pdf"),dpi=300,bbox_inches="tight")


if __name__ == "__main__":
    main()