import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.lines as lines

from python.utils.utils import plot_heatmaps

def main():
    """
    Plot Fig. 10 containing heatmaps. Note that this takes a while.

    Saves plots to output/figs
    """

    # For saving
    save_dir = os.path.join("output", "figs")
    os.makedirs(save_dir, exist_ok=True)

    # Set for plotting
    mpl.rc('font', family='Arial')

    # Read fixation data from CSV files
    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))
    
    try:
        ueyes_fixations = pd.read_csv(os.path.join("output", "data", "ueyes_fixations.csv"))
    except:
        print("Did you format UEyes data? Run python.src.pre_process.ueyes scripts.")
        sys.exit()

    # Setup the figure and subplots
    fig, axs = plt.subplots(6, 4, figsize=(10, 15))
    
    # Define category mapping for subplot titles
    category_mapping = {0: "All UI types", 1: "Website", 2: "Desktop UI", 3: "Mobile UI"}

    alphabet_index = 0
    row = 0
    with tqdm(total=24) as pbar:
        for index, data_type in enumerate(["search", "search-absent", "ueyes"]):

            # Configure type of fixation column used based on data source (see python.quick_start.quick_start.ipynb)
            if data_type == "search":
                xcol, ycol = "FPOGX_scaled", "FPOGY_scaled"
                basedata = fixations[fixations.absent == False]
                just_screenshot = True
                normalize = False
                ueyes = False

            elif data_type == "search-absent":
                xcol, ycol = "FPOGX_scaled", "FPOGY_scaled"
                basedata = fixations[fixations.absent == True]
                just_screenshot = True
                normalize = False
                ueyes = False

            elif data_type == "ueyes":
                xcol, ycol = "FPOGX_scaled", "FPOGY_scaled"
                basedata = ueyes_fixations
                just_screenshot = False
                normalize = False
                ueyes = True

            # Plot heatmaps for first and later fixations for each category
            if "search" in data_type or "ueyes" in data_type:
                for first_fixations in [True, False]:
                    for col, category in enumerate(["all", "web", "desktop", "mobile"]):
                        ax = axs[row, col]
                        plot_heatmaps(basedata=basedata, img_type=2, xcol=xcol, ycol=ycol, 
                                      just_screenshot=just_screenshot, normalize=normalize, xmax=1, ymax=1, ueyes=ueyes,
                                      first_fixations=first_fixations, category=category, alphabet_index=alphabet_index, 
                                      ax=ax, debias=False)
                        alphabet_index += 1
                        pbar.update(1)
                    row += 1

    # Add labels to the subplots for fixation types
    for row in [0, 2, 4]:
        axs[row, 0].set_ylabel("Third fixation", size=12)

    for row in [1, 3, 5]:
        axs[row, 0].set_ylabel("Later fixations", size=12)

    # Annotate axes for visual separation
    axs[1, 0].annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1))

    # Set titles for columns in the subplots
    for col in range(4):
        axs[0, col].set_title(category_mapping[col], weight="bold", size=14)
        axs[2, col].set_title(category_mapping[col], weight="bold", size=14)
        axs[4, col].set_title(category_mapping[col], weight="bold", size=14)

    # Add dashed lines to visually separate sections
    fig.add_artist(lines.Line2D([0.05, 0.9], [0.36, 0.36], c="black", linestyle="dashed"))
    fig.add_artist(lines.Line2D([0.05, 0.9], [0.635, 0.635], c="black", linestyle="dashed"))

    # Label
    fig.text(-5, 0.7, "Free-viewing (UEyes)", transform=ax.transAxes, size=14, weight='bold', rotation=90, ha='center')
    fig.text(-5, 6.8, "Visual search (ours)\nTarget present", transform=ax.transAxes, size=13, weight="bold", rotation=90, ha='center')
    fig.text(-5, 3.8, "Visual search (ours)\nTarget absent", transform=ax.transAxes, size=13, weight="bold", rotation=90, ha='center')
    fig.text(1.2, 3.65, "Fixations along y-axis", transform=ax.transAxes, size=12, rotation=-90, ha='center')
    fig.text(-1.8, -0.3, "Fixations along x-axis", transform=ax.transAxes, size=12, ha='center')

    # Adjust subplot spacing and save the figure
    plt.subplots_adjust(hspace=0.5)
    fig.savefig(os.path.join(save_dir, "fig10.pdf"), dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
