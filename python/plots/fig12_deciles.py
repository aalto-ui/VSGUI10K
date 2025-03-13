import pandas as pd
import os
import matplotlib as mpl
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib as mpl

def main():
    """
    Plot Fig. 12 for Guess-Scan-Confirm deciles analysis, showing the relationship between fixation deciles and the distance to the target.
    """

    # Set up the saving directory
    save_dir = os.path.join("output", "figs")
    os.makedirs(save_dir, exist_ok=True)

    # Configure plot settings: use colorblind palette and set font to Arial
    palette = "colorblind"
    mpl.rc('font', family='Arial')

    # Load target found data from CSV, exit if not found
    try:
        tgt_found = pd.read_csv(os.path.join("output", "data", "tgt_found_trials.csv"))
    except FileNotFoundError:
        print("Did you format target found data? Run python.src.pre_process.03_get_target_found_trials script.")
        sys.exit()

    # Group data by participant ID and image name, then compute the max fixation identifier (FPOGID_max)
    grouped_tmp = tgt_found.groupby(["pid", "new_img_name"]).agg({"FPOGID": "max"}).reset_index()
    grouped_tmp = grouped_tmp.rename(columns={"FPOGID": "FPOGID_max"})
    tgt_found = tgt_found.merge(grouped_tmp, on=["pid", "new_img_name"])
    
    # Normalize fixation identifier to a percentage scale (0-100)
    tgt_found["FPOGID_norm"] = tgt_found.FPOGID / tgt_found.FPOGID_max * 100
    
    # Bin normalized fixations into 10 deciles
    _, hist = np.histogram(tgt_found.FPOGID_norm, bins=10)
    tgt_found["FPOGID_norm_binned"] = pd.cut(tgt_found['FPOGID_norm'], hist, precision=0)

    # Filter data for present targets and visual search screens
    df = tgt_found[(tgt_found.absent == False) & (tgt_found.img_type == 2)].copy()
    
    # Tag trials based on median trial length
    tag = np.round(df.FPOGID_max.median(), 0)
    df["FPOGID_max_tag"] = f"< {tag} fixations"
    df.loc[df.FPOGID_max >= tag, ["FPOGID_max_tag"]] = f">= {tag} fixations"

    # Compute target distance in pixels
    df["tgt_distance_y_perimeter_px"] = df.tgt_distance_y_perimeter * 1200
    df["tgt_distance_x_perimeter_px"] = df.tgt_distance_x_perimeter * 1920

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.pointplot(df, x="FPOGID_norm_binned", y="tgt_distance_perimeter_px", ax=ax, legend=False, errorbar="ci", 
                  linewidth=2, zorder=10, hue="FPOGID_max_tag", palette=palette)

    # Set axis labels
    ax.set_xlabel("Fixation decile")
    ax.set_ylabel("Distance to target (px)")

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Annotate the plot to indicate direction of movement towards and away from the target
    ax.annotate('Towards\ntarget', xy=(0.8, 0.35), xycoords='axes fraction', xytext=(0.65, 0.55), 
                arrowprops=dict(arrowstyle="->", color='grey', linestyle="--"), ha="center")

    ax.annotate('Away from target', xy=(0.04, 0.8), xycoords='axes fraction', xytext=(0.3, 0.95), 
                arrowprops=dict(arrowstyle="<-", color='grey', linestyle="--"), ha="center")

    # Set y-axis limits
    ax.set_ylim(0, 500)

    # Configure legend with custom labels and handles
    blue_line = Line2D([], [], color=sns.color_palette(palette)[0], marker="s", markersize=5, label=f"< {tag} fixations")
    purple_line = Line2D([], [], color=sns.color_palette(palette)[1], markersize=5, marker="o", label=f">= {tag} fixations")
    handles = [purple_line, blue_line]
    labels = [h.get_label() for h in handles]
    ax.legend(handles=handles, labels=labels, loc="lower left", ncol=1, title="Trial length", fancybox=True, title_fontproperties={"weight": "bold"})

    # Set x-axis tick labels to range from 1 to 10 (deciles)
    ax.set_xticklabels(np.arange(1, 11))

    # Save the figure
    fig.savefig(os.path.join(save_dir, "fig12.pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
