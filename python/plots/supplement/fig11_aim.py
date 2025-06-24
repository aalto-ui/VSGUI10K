import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    """
    Plot Fig 9 from AIM metrics (figure-ground contrast and contour congestion).
    Saves plots to output/figs/supplement
    """

    # Configure directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Retrieve data
    results = pd.read_csv(os.path.join("output", "data", "vsgui10k_aim_results_with_factors.csv"))

    # Columns pertaining to figure-ground contrast and contour congestion, refer to AIM
    other_cols  =['m5_0', 'm6_0'] 
    other_labels=["Figure-ground contrast", "Contour congestion (proportion)"]

    other_colors = [0,0]
    other_palette= ["#D35FB7"]

    # Create dictionaries of the metrics
    other_dict = {k : i for k,i in zip(other_cols, other_labels)}

    fig, axs = plt.subplots(1,2, figsize = (6, 2))

    for index, metric in enumerate(other_cols):
        ax = axs[index]

        sns.histplot(data=results, x=metric, ax=ax, bins=20, color=other_palette[other_colors[index]], stat="percent")
        ax.set_xlabel(other_dict[metric])

        print (f"{other_dict[metric]}: [{np.around(results[metric].min(), 2)}, {np.around(results[metric].max(), 2)}]")


        ax.set_ylim(0,20)

        if index > 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
        else:
            ax.set_ylabel("Percentage")

    plt.subplots_adjust(wspace=0.1, hspace=0.8)

    plt.savefig(os.path.join(save_dir, f"supp_fig11.pdf"),dpi=300,bbox_inches="tight")

if __name__ == "__main__":
    main()