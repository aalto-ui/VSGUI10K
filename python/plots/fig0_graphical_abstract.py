import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
from PIL import Image
from matplotlib.patches import FancyArrowPatch

from python.utils.utils import plot_heatmap_scipy

def main():
    """
    Plot Fig. 0 graphical abstract.
    
    Saves plots to output/figs
    """

    save_dir = os.path.join("output", "figs") # Save files here 
    os.makedirs(save_dir, exist_ok=True)
    task_dir = os.path.join("images", "task-image") # Retrieve the "experiment" image, Fig. 2

    mpl.rc('font',family='Arial') # Set font

    ##############
    ## Get data ##
    ##############

    fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))
    fixations = fixations[fixations.img_type==2] # Filter only visual search screens

    # Add maximum number of fixations for filtering
    grouped = fixations.groupby(["new_img_name"]).agg({"FPOGID" :"max", "absent" : "mean", "category" : "count"}).reset_index()
    grouped = grouped.rename({"FPOGID" : "FPOGID_MAX"}, axis="columns")
    fixations = fixations.merge(grouped[["new_img_name","FPOGID_MAX"]], how="left", on="new_img_name")
    fixations["FPOGID_NORM"] = (fixations["FPOGID"]) /  (fixations["FPOGID_MAX"])

    # Filter to only trials longer than six fixations
    fixations = fixations[fixations.FPOGID_MAX>=7]

    ##########
    ## Plot ##
    ########## 

    # Define the mosaic layout
    fig = plt.figure(layout="constrained", figsize=(7,8))

    axs = fig.subplot_mosaic(
        """
        AAA
        BCD
        EFG
        """, height_ratios=[2.5,1,1.5],
        gridspec_kw={
            "wspace": 0.01,
            "hspace": 0.25,
        },
    )

    # Load trial image and display
    image_path = os.path.join(task_dir, "fig2-trial.png")
    image = Image.open(image_path)
    axs["A"].imshow(image)
    axs["A"].axis('off') # Hide the axis spines

    ##################################
    ## Plot Factor1 vs. Search Time ##
    ##################################

    ax = axs["D"]

    # This data is obtained from the R script <3-interaction.R>
    f1 = np.array([-2,-1,0,1,2,3,4])
    predicted_present = np.array([2.95,3.34,3.73,4.12,4.51,4.90,5.29]) 
    predicted_absent = np.array([8.96,9.61,10.26,10.91,11.56,12.21,12.86])

    CI_lower_bounds_present = np.array([2.47,2.95,3.39,3.77,4.09,4.38,4.65])
    CI_upper_bounds_present = np.array([3.43,3.73,4.06,4.46,4.92,5.41,5.92])

    CI_lower_bounds_absent = np.array([7.97,8.85,9.61,10.18,10.62,10.99,11.32])
    CI_upper_bounds_absent = np.array([9.95,10.37,10.91,11.63,12.49,13.43,14.39])

    # Plot errors
    ax.fill_between(f1, CI_lower_bounds_present, CI_upper_bounds_present, alpha=.3, color="#1f449C")
    ax.fill_between(f1, CI_lower_bounds_absent, CI_upper_bounds_absent, alpha=.3, color="#f05039")

    # Plot data
    ax.plot(f1, predicted_present, c="#1f449C")
    ax.scatter(f1, predicted_present, c="#1f449C", marker = "^", label="Target present")

    ax.plot(f1, predicted_absent, c="#f05039")
    ax.scatter(f1, predicted_absent, c="#f05039", label = "Target absent")

    # Set labels
    ax.set_ylabel("", size=13)
    ax.set_xlabel("")

    ###################################
    ## Plot Category vs. Search Time ##
    ###################################

    ax = axs["B"]

    category = ["Mobile UI", "Mobile UI", "Desktop UI", "Desktop UI", "Webpage", "Webpage"]
    absence = ["Target present", "Target absent","Target present", "Target absent", "Target present", "Target absent"]
    
    # This data is obtained from the R script <3-interaction.R>
    predicted = [2.77, 7.94, 3.85, 9.86, 3.74, 10.27] 
    cat_data=pd.DataFrame({"Category" : category, "absence" : absence, "predicted" : predicted})

    errors = np.array(predicted) - np.array([2.44, 7.29, 3.53, 9.28, 3.4, 9.62]) # Compute errors

    # Set palette
    palette = ["#5BA300","#ccff66","#0073E6"]

    # Plot data 
    sns.barplot(data=cat_data, x="absence", y="predicted", hue="Category", ax=ax, palette=palette)

    # Plot errors
    x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=np.pad(errors,(0,3)), fmt="none", c= "k")

    # Set labels
    ax.set_xlabel("", size=13)
    ax.set_ylabel("")

    ##############################
    ## Plot Cue vs. Search Time ##
    ##############################

    ax = axs["C"]

    cue = ["Image", "Image", "Text", "Text", "Text+color", "Text+color"]
 
    predicted_cue = [3.74, 10.27, 4.90, 13.45, 5.04, 12.04] 
    cue_data = pd.DataFrame({"Cue type" : cue, "absence" : absence, "predicted" : predicted_cue})

    palette = ["#0073E6", "#f4a4cc", "#b51963"]

    # Plot data
    sns.barplot(data=cue_data, x="absence", y="predicted", hue="Cue type", ax=ax, palette=palette)

    errors_cue = np.array(predicted_cue) - np.array([3.40, 9.62, 4.55, 12.71, 4.69, 11.33])

    # Plot errors
    x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=np.pad(errors_cue,(0,3)), fmt="none", c= "k")

    # Set labels
    ax.set_ylabel("")
    ax.set_xlabel("", size=13)
    axs["B"].set_ylabel("Predicted\nsearch time (s)")

    letters = ["B", "C", "D"]
    for index in range(3):
        ax = axs[letters[index]]

        ax.set_ylim(0,18)
        if index >= 2:
            ax.legend(loc = "upper left", ncol=1, fontsize = "x-small")

        elif index <=1:
            handles, labels = ax.get_legend_handles_labels()
            if "Image" in labels:
                labels[0] = "Image*"
            else:
                labels[2] = "Webpage*"

            ax.legend(handles, labels, loc = "upper left", ncol=1, fontsize = "x-small")

    ######################
    ## Plot adjustments ##
    ######################

    # Display baselines for each variable
    Factor1 =  0.03
    category =  "Web"
    cue = "Image"

    ticks = [[-2, Factor1, 4]]
    labels = [["-2", r"0.03$^*$", "4"]]

    for index in range(2,3):
        ax = axs[letters[index]]
        ax.set_xticks(ticks[index-2])
        ax.set_xticklabels(labels[index-2], rotation=0, fontsize=8)

    # Set labels
    axs["B"].text(-0.75,-10, r"$^*$Reference level", fontsize=10, fontstyle="italic")
    axs["D"].text(-2.2,-6, "Visual clutter (factor score)", fontsize=11)

    #########################
    ## Inform-Scan-Confirm ##
    #########################

    # Take a sample from target at lower-right
    sample = fixations[(fixations.absent == False) & (fixations.tgt_location == "lower-right")] 

    # Plot Guess
    ax = axs["E"]
    phase_sample = sample[(sample["FPOGID"] >= 3) & (sample["FPOGID"] < 5) ]
    plot_heatmap_scipy(phase_sample, ax,  xcol= "FPOGX_scaled", ycol = "FPOGY_scaled", scatter=False, return_density_max=False, xmax=1, ymax=1)
    ax.set_xlabel("Guess", size=13,  weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Scan
    ax = axs["F"]
    phase_sample = sample[(sample["FPOGID"] >= 5) & (sample["FPOGID"] < sample.FPOGID_MAX - 2) ]
    plot_heatmap_scipy(phase_sample, ax,  xcol= "FPOGX_scaled", ycol = "FPOGY_scaled", scatter=False, return_density_max=False, xmax=1, ymax=1)
    ax.set_xlabel("Scan", size=13,  weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Confirm
    ax = axs["G"]
    phase_sample = sample[(sample["FPOGID"] >= sample.FPOGID_MAX - 2)]
    plot_heatmap_scipy(phase_sample, ax,  xcol= "FPOGX_scaled", ycol = "FPOGY_scaled", scatter=False, return_density_max=False, xmax=1, ymax=1)
    ax.set_xlabel("Confirm", size=13, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw target box
    for ax in [axs["E"], axs["F"], axs["G"]]:
        ymin = 0.52
        ymax=1
        alignment = "top"
        y = 0.52
        xmin = 0.52
        xmax = 1
        x = 0.52

        ax.vlines(x = x, ymin = ymin, ymax = ymax, color = "black", linestyle = "dashed")
        ax.hlines(y = y, xmin = xmin, xmax = xmax, color = "black", linestyle = "dashed")

        ax.text(x = xmin + 0.25, y = ymin + 0.2, s = f"Target\nlocation", 
                                horizontalalignment = "center", verticalalignment = alignment)

    plt.annotate('', xy=(0.02, 0.65), xycoords='figure fraction', xytext=(1, 0.65), 
                arrowprops=dict(arrowstyle="-", color='black', linestyle="--"), ha="center")

    plt.annotate('', xy=(0.02, 0.33), xycoords='figure fraction', xytext=(1, 0.33), 
                arrowprops=dict(arrowstyle="-", color='black', linestyle="-"), ha="center")

    # Draw arrows
    for start, end in zip((0.41, 0.73), (0.35, 0.67)):
        start_data = (start, 0.15)
        end_data = (end, 0.15)
        arrow = FancyArrowPatch(start_data, end_data,
                                transform=fig.transFigure,  # Use the figure's transformation
                                connectionstyle="arc3,rad=0.3",  # Optional: Make the arrow curved
                                arrowstyle="<-,head_width=0.05,head_length=0.1",  # Arrow style
                                mutation_scale=40,  # Size of the arrow head'
                                color="black")  # Color of the arrow

        fig.patches.append(arrow)
        
    ###################
    ## Adjust titles ##
    ###################

    fig.text(0.02,1, "1: We study visual search in naturalistic GUIs in an eye-tracking study "+r"($N=84$).", size=13, weight = "bold")
    fig.text(0.02,0.58, "2: The impact of GUI and cue type, absence of the target and visual complexity\non search times is quantified.", size=13, weight = "bold")
    fig.text(0.02,0.28, "3: Our results are summarized in a three-stage pattern of visual search.", size=13, weight = "bold")
    fig.text(0.27,0.67, "One trial in\na controlled study", size=10, weight = "regular", style="italic", horizontalalignment="center")

    # Save plot
    plt.savefig(os.path.join(save_dir, f"fig0.pdf"),dpi=300,bbox_inches="tight")

if __name__ == "__main__":
    main()