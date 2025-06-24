import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from python.utils.utils import draw_scanpath

def main():
    """
    Plot Fig. 27 containing scanpaths.

    Saves plots to output/figs/supplement
    """
    # Configure directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Set up matplotlib
    mpl.rc('font',family='Sans-serif')

    # Get fixations and UEyes data
    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))
    fixations = fixations[fixations.img_type==2] # Filter only visual search trials
    ueyes_fixations = pd.read_csv(os.path.join("output", "data", "ueyes_fixations.csv"))

    # Find example
    absent_images = np.unique(fixations[fixations.absent == True].img_name)

    # Set up plot
    fig,axs = plt.subplots(6,3, figsize=(11,10))
    title_dict = {1: "B: Visual Search\n(Target Present)", 
                2: "C: Visual Search\n(Target Absent)", 
                0: "A: Free viewing\n(No Target)"}


    categories = ["web", "web", "desktop", "desktop", "mobile", "mobile"]

    # Plot
    for row in range(6):
        np.random.seed(122+row) # Set seed for replication

        # Choose image
        random_img = np.random.choice(np.unique(fixations[(fixations.img_name.isin(absent_images)) & (fixations.category == categories[row])].img_name))

        # Get samples for the chosen image
        fixation_sample_absent = fixations[(fixations.img_name == random_img) & (fixations.absent == True)]
        fixation_sample_absent = fixation_sample_absent[fixation_sample_absent.pid == np.random.choice(np.unique(fixation_sample_absent.pid))]

        fixation_sample_present = fixations[(fixations.img_name == random_img) & (fixations.absent == False)]
        fixation_sample_present = fixation_sample_present[fixation_sample_present.pid == np.random.choice(np.unique(fixation_sample_present.pid))]

        ueyes_sample = ueyes_fixations[ueyes_fixations.img_name == random_img][0:11]

        # Get scaling factors for plotting the fixations correctly
        scale_x = np.array(fixation_sample_present.scale_x)[0]
        scale_y = np.array(fixation_sample_present.scale_y)[0]

        media_id = str(np.unique(fixation_sample_present.img_name)[0])

        original_width = np.array(fixation_sample_present.original_width)[0]
        original_height = np.array(fixation_sample_present.original_height)[0]
        
        # Plot each pane for visual search (target present, target absent) and free-viewing
        for index, fixations_sample in enumerate([ueyes_sample, fixation_sample_present, fixation_sample_absent]):   
            
            ax = axs[row,index]

            if index == 0: # Free viewing
                xs = fixations_sample.FPOGX_scaled.to_numpy() 
                ys = fixations_sample.FPOGY_scaled.to_numpy() 

            
            else: # Visual search
                xs = (fixations_sample.FPOGX_debias.to_numpy() - (1-scale_x) / 2) / scale_x
                ys = (fixations_sample.FPOGY_debias.to_numpy() - (1-scale_y) / 2) / scale_y
            
            # Get durations and draw scanpath
            ts = fixations_sample.FPOGD.to_numpy()
            
            ax.set_ylim(original_height,0)
            ax.set_xlim(0,original_width)

            img_path = os.path.join("data", "vsgui10k-images", media_id)
            _img = draw_scanpath(xs, ys, ts, img_path, ax) 
                
            ax.set_yticks([])
            ax.set_xticks([])
            if row == 0:
                ax.set_title(title_dict[index])

    plt.subplots_adjust(
                        wspace=0, 
                        hspace=0.05)
    
    fig.savefig(os.path.join(save_dir, f"supp_fig27.pdf"),dpi=300,bbox_inches="tight", pad_inches=0.2)

if __name__ == "__main__":
    main()