# Note that you need to run the R script for factor scores <fa.R> before this plot

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def main():
    """
    Plot Fig. 6 demonstrating factor scores.

    Saves plots to output/figs
    """
    
    # For images and saving
    files_dir = os.path.join("data", "vsgui10k-images")
    save_dir = os.path.join("output", "figs")
    os.makedirs(save_dir, exist_ok=True)

    # Load results data with factor scores
    try:
        results = pd.read_csv(os.path.join("output", "data", "vsgui10k_aim_results_with_factors.csv"))
    except:
        print("Did you run factor analysis? Run R/fa.R script.")
        sys.exit()

    # Initialize the figure and subplots
    fig, axs = plt.subplots(4, 3, figsize=(6, 7))

    # Define metrics columns and reduced set for plotting
    metrics_cols = ['Factor1', 'Factor2', 'Factor3', 'Factor4', 'm5_0', 'm6_0']
    metrics_cols_reduced = ['m6_0', 'Factor1', 'Factor4']

    # Define palette and specific colors for metrics
    metrics_palette = ["#D81B60", "#1E88E5", "#FFC107", "#004D40", "#D35FB7"]
    metrics_colors = [4, 0, 1]

    # Define labels for metrics
    metrics_labels = ["Factor 1\n(Visual clutter)", "Factor 2\n(Color variability)", "Factor 3\n(Color variability)", "Factor 4\n(Grid quality)", "Figure-ground\ncontrast", "Contour\ncongestion (%)"]
    metrics_dict = {k: i for k, i in zip(metrics_cols, metrics_labels)}

    # Set random seed for reproducibility (only change if different images should be plotted)
    np.random.seed(123456)

    # Iterate over selected metrics and rows for plot creation
    cols = metrics_cols_reduced
    for col, metric in enumerate(cols):
        for row in range(4):
            ax = axs[row, col]
            if row == 0:
                # Plot histogram for the metric
                sns.histplot(data=results, x=metric, ax=ax, bins=20, color=metrics_palette[metrics_colors[col]], stat="percent", zorder=10)
                ax.set_title(metrics_dict[metric], y=1.4, size=10, fontweight="bold")
                ax.set_ylabel("")
                ax.set_xlabel("")
                
                ax.set_ylim(0, 20)
                
                if col != 0:
                    ax.set_yticks([])
                
                if col == 0:
                    ax.set_ylabel("Percentage\n(over all 900 images)")
            else:
                # Select sample images based on metric value ranges, Low, Medium, High
                if row == 1:
                    sample = np.unique(results[results[metric] < results[metric].mean() - results[metric].std() / 2].img_name)
                    string = "Low"
                elif row == 2:
                    sample = np.unique(results[(results[metric] >= results[metric].mean() - results[metric].std() / 2) & (results[metric] <= results[metric].mean() + results[metric].std() / 3)].img_name)
                    string = "Medium"
                elif row == 3:
                    sample = np.unique(results[results[metric] > results[metric].mean() + results[metric].std() / 2].img_name)
                    string = "High"

                # Randomly select a sample image
                img_name = np.random.choice(sample)
                val = results[results["img_name"] == img_name][metric].item()

                # Draw dashed line on histogram for sample value
                axs[0, col].vlines(x=val, ymin=0, ymax=20, linestyles="dashed", color="black", alpha=0.5)
                axs[0, col].text(val, 20.5, string, horizontalalignment='left', fontstyle="italic", rotation=65, size=8)

                # Load and display the selected sample image
                path = os.path.join(files_dir, img_name)
                try:
                    image = plt.imread(path)
                except:
                    image = plt.imread(path, format="jpeg")
                ax.imshow(image)

                ax.set_xticks([])
                ax.set_yticks([])

    # Add annotations for Low, Medium, and High categories
    for string, x in zip(["Low", "Medium", "High"], [2.85, 1.7, 0.5]):
        plt.text(-3.6, x, string, color='black', fontstyle="italic", size=10,
                 bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'), transform=ax.transAxes, horizontalalignment="center")

    # Adjust layout for final figure
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    # Save the figure to PDF file
    plt.savefig(os.path.join(save_dir, f"fig6.pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
