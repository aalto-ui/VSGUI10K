import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
import matplotlib as mpl

def main():
    """
    Plot Fig. 26 containing saccade directions.

    Saves plots to output/figs/supplement
    """
    # Configure directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Configure matplotlib
    mpl.rc('font',family='Arial')
    colors = ["#7ca1cc", "#1f449C"]

    # Retrieve fixations
    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))
    fixations = fixations[fixations.img_type == 2] # Filter only visual search trials

    # Set up polar histogram
    fig = plt.figure(figsize=(6,16))

    ax0 = fig.add_subplot(521, projection='polar')
    ax1 = fig.add_subplot(522, projection='polar')
    ax2 = fig.add_subplot(523, projection='polar')
    ax3 = fig.add_subplot(524, projection='polar')
    ax4 = fig.add_subplot(525, projection='polar')
    ax5 = fig.add_subplot(526, projection='polar')
    ax6 = fig.add_subplot(527, projection='polar')
    ax7 = fig.add_subplot(528, projection='polar')
    ax8 = fig.add_subplot(529, projection='polar')
    ax9 = fig.add_subplot(5, 2, 10, projection='polar')


    axs = [ax0, ax2, ax4, ax6, ax8, ax1, ax3, ax5, ax7, ax9]

    # Plot
    index = 0
    for col, first in enumerate([True, False]):
        for tgt_idx, tgt_loc in enumerate(["upper-left", "lower-left", "upper-right", "lower-right", "absent"]):

            ax = axs[index]
            
            fixation_sample = fixations # Reset fixations
        
            fixation_sample = fixation_sample[fixation_sample.tgt_location == tgt_loc] # Get fixations for particular target location
            
            if first:
                fixation_sample = fixation_sample[(fixation_sample.FPOGID < 4)] # Filter first fixations if needed
            else:
                fixation_sample = fixation_sample[(fixation_sample.FPOGID >= 4)]
                
            
            degrees = np.array(fixation_sample.SACCADE_DIR) # Retrieve saccade direction data

            bin_size = 20
            a, b=np.histogram(degrees, bins=np.arange(0, 360+bin_size, bin_size))
            centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

            ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color=f"{colors[1]}", edgecolor='k', alpha=0.7)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            if col == 0:
                if tgt_loc == "absent":
                    ax.set_ylabel(f"Target {tgt_loc}", size=14, weight="bold", x=-0.5)
                    ax.yaxis.set_label_coords(-0.25, 0.5)
                else:
                    ax.set_ylabel(f"Target at\n{tgt_loc}", size=14, weight="bold", x=-0.5)
                    ax.yaxis.set_label_coords(-.25, 0.5)

                                        
            ax.text(-0.1, 1.1, string.ascii_uppercase[index], transform=ax.transAxes, 
                    size=14, weight='bold')
            
            ax.set_yticks([])
            index +=1

    ax0.set_title("First fixations\n(1st-3rd)", size=12, y = 1.1)
    ax1.set_title("Later fixations\n(after 3rd)", size=12, y = 1.1)
    fig.text(-0.35, -0.25, "Saccade directions (degrees)", transform=ax.transAxes, size=12, ha='center')
    ax0.yaxis.set_label_coords(-0.3,0.5)
    ax5.yaxis.set_label_coords(-0.3,0.5)
    plt.subplots_adjust(wspace=0.5, hspace=-0.1)

    fig.savefig(os.path.join(save_dir, "supp_fig26.pdf"), dpi=300, bbox_inches = "tight")   

if __name__ == "__main__":
    main()