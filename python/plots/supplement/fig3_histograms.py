import matplotlib as mpl
import os 
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

def main():
    
    # Define directories for saving the figure
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)
    
    # Configure Matplotlib 
    mpl.rc('font', family='Arial')
    
    # Read search times data from a CSV file
    search_times = pd.read_csv(os.path.join("output", "data", "vsgui10k_search_times.csv"))
    
    # Define mapping for UI categories
    category_mapping = {1: "All UI types", 2: "Website", 1: "Desktop UI", 0: "Mobile UI"}
    
    # Create subplots with 2 rows and 3 columns, each subplot of size 6x4 inches
    fig, axs = plt.subplots(2, 3, figsize=(6, 4))
    palette = ["#5BA300", "#ccff66", "#0073E6"]
    
    alphabet_index = 0
    # Iterate through 'False' and 'True' for target absent and present conditions
    for row, absent in enumerate([False, True]):
        for col, category in enumerate(["mobile", "desktop", "web"]):
            sample = search_times[search_times.absent == absent]
            if category != "all":
                sample = sample[sample.category == category]
            ax = axs[row, col]
            
            # Plot histogram of search time for the defined sample
            sns.histplot(
                data=sample,
                x="search_time",
                stat="percent",
                ax=ax,
                bins=20,
                color=palette[col]
            )
            ax.set_title(category)
            ax.set_ylim(0, 50)
            ax.set_xlim(-1, 30)
            
            # Set titles for subplots
            if row == 0:
                ax.set_title(category_mapping[col], weight="bold")
            else:
                ax.set_title("")
                
            # Remove x and y axis labels
            ax.set_ylabel("")
            ax.set_xlabel("")
            
            # Annotate subplot with the appropriate letter
            ax.text(0, 1.05, string.ascii_uppercase[alphabet_index], transform=ax.transAxes,
                    size=12, weight='bold')
            alphabet_index += 1
            
            # Set custom y and x axis tick positions
            ax.set_yticks([10, 20, 30, 40, 50])
            ax.set_xticks([10, 20, 30])
    
    # Add text labels to indicate 'Target present' and 'Target absent' conditions
    fig.text(1.1, 1.35, "Target present", transform=ax.transAxes,
             size=12, weight='bold', rotation=-90, ha='center')
    fig.text(1.1, 0.18, "Target absent", transform=ax.transAxes,
             size=12, weight='bold', rotation=-90, ha='center')
    
    # Set x and y axis labels for the second row, first column subplot
    axs[1, 1].set_xlabel("Search time (s)", size=12)
    axs[1, 0].set_ylabel("Percent of trials", y=1.1, x=-0.3, rotation=90, size=12)
    
    # Save the figure as a PDF file in the specified directory
    fig.savefig(os.path.join(save_dir, "supp_fig3.pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()