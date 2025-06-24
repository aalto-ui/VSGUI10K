import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

def main():
    """
    Plot Fig 25 for Guess-Scan-Confirm.
    Saves plots to output/figs/supplement
    """
    # Configure folders for plotting
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Configure matplotlib
    mpl.rc('font',family='Arial')
    palette = "Set2"

    # Get data
    tgt_found = pd.read_csv(os.path.join("output", "data", "tgt_found_trials.csv"))

    # Add normalized duration
    grouped_tmp = tgt_found.groupby(["pid", "new_img_name"]).agg({"FPOGID" : "max"}).reset_index()
    grouped_tmp = grouped_tmp.rename(columns={"FPOGID": "FPOGID_max"})
    tgt_found= tgt_found.merge(grouped_tmp[["pid", "new_img_name", "FPOGID_max"]], on = ["pid", "new_img_name"])

    tgt_found["FPOGID_norm"] = tgt_found.FPOGID / tgt_found.FPOGID_max * 100

    # Bin normalized duration
    bins, hist = np.histogram(tgt_found.FPOGID_norm, bins=10)

    tgt_found["FPOGID_norm_binned"] = pd.cut(tgt_found['FPOGID_norm'], hist, precision=0)

    # Filter only to visual search screens
    df = tgt_found[(tgt_found.absent == False) & (tgt_found.img_type == 2)]

    # Add X and Y distance to target
    df["tgt_distance_x_perimeter_px"] = df.tgt_distance_x_perimeter * 1920
    df["tgt_distance_y_perimeter_px"] = df.tgt_distance_y_perimeter * 1200

    # Plot
    fig ,axs = plt.subplots(2,2, figsize=(9,7))

    tgt_locations = np.array([["upper-left", "upper-right"], ["lower-left", "lower-right"]])

    for row in range(2):
        for col in range(2):

            ax = axs[row,col]

            loc = tgt_locations[row,col]
            df_tmp = df[df.tgt_location == loc]

            # Add arrows
            ax.annotate('Towards\ntarget', xy=(0.8, 0.15), xycoords='axes fraction', xytext=(0.65, 0.25), 
                arrowprops=dict(arrowstyle="->", color='grey', linestyle="--"), ha="center")
            ax.annotate('Away from target', xy=(0.04, 0.8), xycoords='axes fraction', xytext=(0.25, 0.9), 
                        arrowprops=dict(arrowstyle="<-", color='grey', linestyle="--"), ha="center")

            # Plot for each bin 
            sns.pointplot(df_tmp, x="FPOGID_norm_binned", y=  "tgt_distance_x_perimeter_px", ax=ax, legend=False, errorbar="ci", linewidth=2, markers="s", color=sns.color_palette(palette)[0])
            sns.pointplot(df_tmp, x="FPOGID_norm_binned", y=  "tgt_distance_y_perimeter_px", ax=ax, errorbar="ci", linewidth=2, markers="o", color=sns.color_palette(palette)[1])

            if row == 1:
                ax.set_xlabel("Fixation decile (%)")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("Mean distance to target (px)")
            else:
                ax.set_ylabel("")

            ax.set_ylim(0,400)

            ins = inset_axes(ax,loc="upper right", width=0.75, height=0.75)
            #ins.set_aspect('equal')
            ins.set_xticks([])
            ins.set_yticks([])

            ins.set_xlim(0,1)
            ins.set_ylim(0,1)

            ax.set_title(f"Target at {loc}", weight="bold")

            fixation_x = 0.45
            fixation_y = 0.9
            tgt_width = 0.2
            tgt_height = 0.2
            ins.text(fixation_x-0.05, fixation_y+0.05,"Fixation", horizontalalignment="center", fontweight="light", fontstyle="italic", size=9, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0'))
            ins.plot([fixation_x], [fixation_y], marker = ".", color="black", zorder=10)

            # Add target location squares
            if loc == "upper-left":

                # Add upper-left

                xy = (0,0.52)
                # "Target"

                tgt_x = 0.1
                tgt_y = 0.53

            if loc == "upper-right":
                # Add upper-right

                xy = (0.52,0.52)

                tgt_x = 0.8
                tgt_y = 0.6

            if loc == "lower-left":
                # Add lower-left

                xy = (0,0)

                tgt_x = 0.1
                tgt_y = 0.1

            if loc == "lower-right":
                # Add lower-right
                xy = (0.52,0)
                tgt_x = 0.8
                tgt_y = 0.1

            rect = Rectangle(xy, 0.48, 0.48, edgecolor="grey", facecolor="white", linestyle="--")
            ins.add_patch(rect)

            rect = Rectangle((tgt_x,tgt_y), tgt_width, tgt_height, edgecolor="black", facecolor="white")
            ins.add_patch(rect)

            if "right" in loc:
                tgt_width = 0
            ins.annotate('', xy=(tgt_x+tgt_width, fixation_y), xycoords='axes fraction', xytext=(fixation_x, fixation_y), 
            arrowprops=dict(arrowstyle="->", color=sns.color_palette(palette)[0], linestyle="-"), ha="center")

            ins.annotate('', xy=(fixation_x,tgt_y+tgt_height), xycoords='axes fraction', xytext=(fixation_x, fixation_y), 
            arrowprops=dict(arrowstyle="->", color=sns.color_palette(palette)[1], linestyle="-"), ha="center")
            ins.text(tgt_x+0.05, tgt_y+0.05,"T", horizontalalignment="left", fontweight="light", fontstyle="italic", size=9)

            if row == 0:
                ax.set_xticklabels([], rotation=45, ha='right')     
            else:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


            blue_line = Line2D([], [], color=sns.color_palette(palette)[0], marker = "s", markersize=5, label='Horizontal distance (x-axis)')
            purple_line = Line2D([], [], color=sns.color_palette(palette)[1], markersize=5, marker="o", label='Vertical distance (y-axis)')

            handles = [blue_line,purple_line]
            labels = [h.get_label() for h in handles] 

            ax.legend(handles=handles, labels=labels, loc = "lower left", ncol=1, fontsize = "x-small")  
    
    # Save
    fig.savefig(os.path.join(save_dir, "supp_fig25.pdf"), dpi=300, bbox_inches = "tight")

if __name__ == "__main__":
    main()