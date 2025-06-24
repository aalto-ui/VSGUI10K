import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import matplotlib as mpl

def main():
    """
    Plot Fig 12 for errors.

    Saves plots to output/figs/supplement
    """

    # Configure directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Set matplotlib configuration
    sns.set_palette(sns.color_palette("colorblind"))
    mpl.rc('font',family='Arial')

    # Retrieve data
    tgt_found = pd.read_csv(os.path.join("output", "data", "tgt_found_trials.csv"))

    print ("Proportion of trials where distance to target perimeter <0.2 normalized distance:",len((tgt_found[(tgt_found.tgt_distance_perimeter <= 0.2) & (tgt_found.img_type==3)])) / len(tgt_found[tgt_found.img_type==3]))

    # Plot
    fig, axs = plt.subplots(2,2,figsize=(6,4))

    titles = ["Distance (normalized)", "Distance (px)"]
    alphabet_index = 0

    datas = {}

    for row, absent in enumerate([False, True]):
        
        tgt_found_validation = tgt_found[(tgt_found.img_type == 3) & (tgt_found.absent == absent)].copy() # Filter to validation screens (3) only
        
        if absent == True:
            tgt_found_validation = tgt_found_validation[tgt_found_validation.FPOGID > 2] # Fixations may trail from previous screen so filter out first fixations for target absent trials
        
        tgt_found_validation_grouped = tgt_found_validation.groupby(["pid", "new_img_name"]).agg({
                                                                        'tgt_distance_px' : "mean", 
                                                                        'tgt_distance' : "mean",
                                                                        'tgt_distance_perimeter' : "mean", 
                                                                        'tgt_distance_perimeter_px' : "mean"}).reset_index().drop_duplicates("new_img_name")
        
        datas[absent] = tgt_found_validation_grouped
            
        # Plot both normalized and pixel distance to target 
        for index, col in enumerate(["tgt_distance_perimeter", "tgt_distance_perimeter_px"]):
            ax = axs[row, index]
            #ax.set_ylim(0,40)
            if index == 1:
                ax.set_xlim(0,1500)

            else:
                
                ax.set_xlim(0,1)
            sns.histplot(data=tgt_found_validation_grouped, x=col, ax=ax, stat="count", element="step", color="lightblue")
            ax.set_xlabel("")
            ax.set_ylabel("")
            
            ax.text(-0.1, 1.08, string.ascii_uppercase[alphabet_index], transform=ax.transAxes, 
                    size=13, weight='bold')
            alphabet_index += 1
            
        for col in range(2):
            axs[1, col].set_xlabel(titles[col], size =12)
            
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Set labels
    axs[1, 0].set_ylabel("Number of trials", y = 1.1, x=-0.3, rotation = 90, size =12)
    fig.text(1.1, 1.55, "Target present", transform=ax.transAxes, 
                        size=12, weight='bold', rotation =-90, ha='center')

    fig.text(1.1, 0.15, "Target absent", transform=ax.transAxes, 
                        size=12, weight='bold', rotation =-90, ha='center')

    fig.savefig(os.path.join(save_dir, "supp_fig12.pdf"), dpi=300, bbox_inches = "tight")

if __name__ == "__main__":
    main()