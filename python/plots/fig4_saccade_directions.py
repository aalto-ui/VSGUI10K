import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
import matplotlib as mpl

def main():
    """
    Plot Fig. 4 for saccade directions per GUI category.

    Saves plots to output/figs
    """
    # Set for saving
    save_dir = os.path.join("output", "figs")
    os.makedirs(save_dir, exist_ok=True)

    # Set for plotting
    mpl.rc('font',family='Arial')

    ##############
    ## Get data ##
    ##############

    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))
    fixations = fixations[fixations.img_type == 2] # Only visual search data
    
    ##########
    ## Plot ##
    ##########

    fig = plt.figure(figsize=(10,6))

    # Add axes with polar projections
    ax0 = fig.add_subplot(241, projection='polar')
    ax1 = fig.add_subplot(242, projection='polar')
    ax2 = fig.add_subplot(243, projection='polar')
 
    # Set colors
    palette = ["#5BA300","#ccff66","#0073E6"]

    # Map category strings for titles
    category_mapping = {"web" : "Website", "desktop" : "Desktop UI", "mobile" : "Mobile UI"}

    axs = [ax0, ax1, ax2]
    index = 0
    for index, category in enumerate(["mobile", "desktop", "web"]):

            ax = axs[index]

            # Filter for this fixation sample
            fixations_sample = fixations[(fixations.category == category)]
                    
            # Get  degrees    
            degrees = np.array(fixations_sample.SACCADE_DIR)

            # Draw polar histogram
            bin_size = 20
            a, b=np.histogram(degrees, bins=np.arange(0, 360+bin_size, bin_size))
            centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])
            ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color=f"{palette[index]}", edgecolor='k', alpha=0.7)
            ax.set_theta_zero_location("E") # Adjust directions
            ax.set_theta_direction(1)
            
            # Set title and lettering
            ax.set_title(category_mapping[category], weight = "bold", size = 14,y=1.16)
                                        
            ax.text(-0.1, 1.1, string.ascii_uppercase[index], transform=ax.transAxes, 
                    size=14, weight='bold')
            
            ax.set_yticks([])
            index +=1

    # Set title and adjust
    fig.text(-1, -0.3, "Saccade directions (degrees)", transform=ax.transAxes, size=12, ha='center')
    ax0.yaxis.set_label_coords(-0.3,0.5)
    plt.subplots_adjust(wspace=0.5, hspace=-0.1)
    
    # Save
    fig.savefig(os.path.join(save_dir, "fig4.pdf"), dpi=300, bbox_inches = "tight")


if __name__ == "__main__":
    main()