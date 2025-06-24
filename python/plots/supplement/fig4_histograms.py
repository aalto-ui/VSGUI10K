import matplotlib as mpl
import os 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import string

def main():

    # Configure directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Configure Matplotlib
    mpl.rc('font',family='Arial')

    # Set number of bins
    num_bins = 20

    # Retrieve data
    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))
    fixations = fixations[fixations.img_type ==2]

    # Map categories
    category_mapping = {1 : "All UI types", 2: "Website", 1 : "Desktop UI", 0 : "Mobile UI"}

    # Create plot
    fig, axs = plt.subplots(2,3, figsize=(6,4))
    palette = ["#5BA300","#ccff66","#0073E6"]

    # Plot 
    alphabet_index = 0
    for row, absent in enumerate([False, True]):
        for col, category in enumerate(["mobile", "desktop", "web"]):
            
            sample = fixations[(fixations.absent == absent)]
            
            if category != "all":
                sample = sample[(sample.category == category)]
            
            ax = axs[row, col]
            
            # Plot histogram of fixation durations
            sns.histplot(data=sample, x="FPOGD",stat="percent", ax =ax,bins=num_bins, color = palette[col])
            ax.set_title(category)

            if row == 0:
                ax.set_title(category_mapping[col], weight="bold")
            else:
                ax.set_title("")
            
            ax.set_ylim(0,70)
            ax.set_ylabel("")
            
            ax.set_xlabel("")
            
            ax.text(-0.15, 1.05, string.ascii_uppercase[alphabet_index], transform=ax.transAxes, 
                        size=12, weight='bold')
            alphabet_index += 1

    # Fix labels      
    fig.text(1.1, 1.35, "Target present", transform=ax.transAxes, 
                        size=12, weight='bold', rotation =-90, ha='center')

    fig.text(1.1, 0.18, "Target absent", transform=ax.transAxes, 
                        size=12, weight='bold', rotation =-90, ha='center')

    # Fix labels
    axs[1, 1].set_xlabel("Fixation duration (s)", size =12)
    axs[1, 0].set_ylabel("Percent of trials", y = 1.1, x=-0.3, rotation = 90, size =12)

    # Save
    fig.savefig(os.path.join(save_dir, "supp_fig4.pdf"), dpi=300, bbox_inches = "tight")

if __name__ == "__main__":
    main()