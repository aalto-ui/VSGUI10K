import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
import sys
import string

def main():
    """
    Plot Fig. 8 showing interactions in the LMER model. The data is obtained from the R script <3-interaction.R>.

    Saves plots to output/figs
    """

    # For saving
    save_dir = os.path.join("output", "figs")
    os.makedirs(save_dir, exist_ok=True)

    # For plotting
    mpl.rc('font',family='Arial')

    # Set up mosaic for plotting
    axs = plt.figure(layout="constrained", figsize=(7,5)).subplot_mosaic(
        """
        AB.
        CDE
        """, 
        gridspec_kw={
            "wspace": 0,
            "hspace": -0.1,
        },
    )

    ##########################
    ## m6_0 vs. Search Time ##
    ##########################

    # This data is obtained from the R script <3-interaction.R>
    m6_0 = np.array([0, 0.25, 0.5, 0.75, 1])
    predicted_present = np.array([2.16, 2.81, 3.46, 4.11, 4.76])
    predicted_absent = np.array([6.40, 8.27, 10.15, 12.02, 13.90])

    CI_lower_bounds_present = np.array([1.19, 2.14, 3.04, 3.74, 4.20])
    CI_upper_bounds_present = np.array([3.14, 3.48, 3.89, 4.48, 5.32])

    CI_lower_bounds_absent = np.array([4.07, 6.69, 9.23, 11.37, 12.79])
    CI_upper_bounds_absent = np.array([8.72, 9.86, 11.07, 12.67, 15.01])

    ax = axs ["C"]

    # Plot errors
    ax.fill_between(m6_0, CI_lower_bounds_present, CI_upper_bounds_present, alpha=.3, color="#1f449C")
    ax.fill_between(m6_0, CI_lower_bounds_absent, CI_upper_bounds_absent, alpha=.3, color="#f05039")

    # Plot data
    ax.plot(m6_0, predicted_present, c="#1f449C")
    ax.scatter(m6_0, predicted_present, c="#1f449C", marker = "^", label="Target present")
    ax.plot(m6_0, predicted_absent, c="#f05039")
    ax.scatter(m6_0, predicted_absent, c="#f05039", label = "Target absent")

    # Set labels
    ax.set_ylabel("Predicted search time (s)", size=13)
    ax.yaxis.set_label_coords(-0.15,1.2)
    ax.set_xlabel("Contour congestion", size=13)

    #############################
    ## Factor1 vs. Search Time ##
    #############################

    # This data is obtained from the R script <3-interaction.R>
    f1 = np.array([-2,-1,0,1,2,3,4])
    predicted_present = np.array([2.93,3.35,3.78,4.20,4.62,5.04,5.46])
    predicted_absent = np.array([9.26,10.16,11.06,11.97,12.87,13.77,14.67])

    CI_lower_bounds_present = np.array([2.41,2.94,3.41,3.82,4.17,4.48,4.77])
    CI_upper_bounds_present = np.array([3.45,3.77,4.14,4.57,5.07,5.60,6.16])

    CI_lower_bounds_absent = np.array([8.19,9.35,10.37,11.20,11.87,12.47, 13.03])
    CI_upper_bounds_absent = np.array([10.33,10.98,11.76,12.74,13.87,15.07,16.31])

    ax = axs["D"]

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
    ax.set_xlabel("Factor 1 (Visual clutter)", size=13, y= 2)

    #############################
    ## Factor4 vs. Search Time ##
    #############################

    # This data is obtained from the R script <3-interaction.R>
    f4 = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5])
    predicted_present = np.array([3.02, 3.28, 3.53, 3.78, 4.03, 4.29, 4.54, 4.79, 5.04])
    predicted_absent = np.array([7.35, 8.59, 9.82, 11.06, 12.30, 13.53, 14.77, 16.00, 17.24])

    CI_lower_bounds_present = np.array([2.49, 2.83, 3.14, 3.42, 3.65, 3.85, 4.02, 4.18, 4.32])
    CI_upper_bounds_present = np.array( [3.56, 3.73, 3.92, 4.14, 4.41, 4.72, 5.05, 5.40, 5.76])

    CI_lower_bounds_absent = np.array([6.18, 7.65, 9.07, 10.37, 11.52, 12.58, 13.57, 14.53, 15.48])
    CI_upper_bounds_absent = np.array([8.52, 9.52, 10.58, 11.75, 13.07, 14.49, 15.97, 17.48, 19.01])

    ax = axs["E"]

    # Plot errors
    ax.fill_between(f4, CI_lower_bounds_present, CI_upper_bounds_present, alpha=.3, color="#1f449C")
    ax.fill_between(f4, CI_lower_bounds_absent, CI_upper_bounds_absent, alpha=.3, color="#f05039")

    # Plot data
    ax.plot(f4, predicted_present, c="#1f449C")
    ax.scatter(f4, predicted_present, c="#1f449C", marker = "^", label="Target present")

    ax.plot(f4, predicted_absent, c="#f05039")
    ax.scatter(f4, predicted_absent, c="#f05039", label = "Target absent")

    # Set labels
    ax.set_ylabel("", size=13)
    ax.set_xlabel("Factor 4 (Grid quality)", size=13)

    ###############################
    ## Category  vs. Search Time ##
    ###############################

    ax = axs["A"]

    category = ["Mobile UI", "Mobile UI", "Desktop UI", "Desktop UI", "Webpage", "Webpage"]
    absence = ["Target present", "Target absent","Target present", "Target absent", "Target present", "Target absent"]

    # This data is obtained from the R script <3-interaction.R>
    predicted = [2.79, 8.25, 3.91, 10.17, 3.79, 11.09]

    cat_data=pd.DataFrame({"Category" : category, "absence" : absence, "predicted" : predicted})
    errors = np.array(predicted) - np.array([2.42, 7.55, 3.56, 9.56, 3.42, 10.39])

    palette = ["#5BA300","#ccff66","#0073E6"]

    # Plot data
    sns.barplot(data=cat_data, x="absence", y="predicted", hue="Category", ax=ax, palette=palette)

    # Plot errors
    x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=np.pad(errors,(0,3)), fmt="none", c= "k")

    # Set labels
    ax.set_ylabel("", size=13)
    ax.set_xlabel("", size=13)

    #########################
    ## Cue vs. Search Time ##
    #########################

    ax = axs["B"]

    cue = ["Image", "Image", "Text", "Text", "Text+color", "Text+color"]

    # This data is obtained from the R script <3-interaction.R>
    predicted_cue = [3.79, 11.09, 5.07, 14.51, 5.17, 13.00]

    cue_data = pd.DataFrame({"Cue type" : cue, "absence" : absence, "predicted" : predicted_cue})

    palette = ["#0073E6", "#f4a4cc", "#b51963"]

    # Plot data
    sns.barplot(data=cue_data, x="absence", y="predicted", hue="Cue type", ax=ax, palette=palette)

    # Plot errors
    errors_cue = np.array(predicted_cue) - np.array([3.42, 10.39, 4.69, 13.73, 4.79, 12.24])

    x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]

    ax.errorbar(x=x_coords, y=y_coords, yerr=np.pad(errors_cue,(0,3)), fmt="none", c= "k")

    # Set labels
    ax.set_xlabel("", size=13)
    ax.set_ylabel("", size=13)

    # Set letters
    letters = ["A", "B", "C", "D", "E"]
    for index in range(5):
        ax = axs[letters[index]]
        ax.set_ylim(0,20)
        if index == 3:
            ax.legend(loc = "upper left", bbox_to_anchor=(0.1, -0.3), ncol=1)

        elif index <=1:
            handles, labels = ax.get_legend_handles_labels()
            if "Image" in labels:
                labels[0] = "Image*"
            else:
                labels[2] = "Webpage*"
            ax.legend(handles, labels, loc = "upper left", ncol=1, fontsize="small")
        ax.text(-0.1, 1.08, string.ascii_uppercase[index], transform=ax.transAxes, 
                        size=15, weight='bold')

    ######################
    ## Plot adjustments ##
    ######################

    Factor1 =  0.03
    Factor4 =  0.02
    m6_0 = 0.63
    category =  "Web"
    cue = "Image"

    axs["C"].vlines(x=m6_0, ymin=0, ymax=20, color="grey", linestyle="dashed",zorder=-10)
    axs["D"].vlines(x=Factor1, ymin=0, ymax=20, color="grey", linestyle="dashed",zorder=-10)
    axs["E"].vlines(x=Factor4, ymin=0, ymax=20, color="grey", linestyle="dashed",zorder=-10)

    ticks = [[0.25, m6_0, 1], [-2, Factor1, 2, 4], [-2.5, Factor4, 2.5, 5]]
    labels = [["0.25", r"0.62$^*$", "1"], ["-2", r"0.02$^*$", "2", "4"], ["-2.5", r"0.02$^*$", "2.5", "5"]]

    for index in range(2,5):
        ax = axs[letters[index]]
        ax.set_xticks(ticks[index-2])
        ax.set_xticklabels(labels[index-2])

    axs["E"].text(0.5,-10, r"$^*$Reference level", fontsize=10, fontstyle="italic")

    # Save
    plt.savefig(os.path.join(save_dir, f"fig8.pdf"),dpi=300,bbox_inches="tight")

if __name__ == "__main__":
    main()
