import os
import pandas as pd
import matplotlib.pyplot as plt
import string
import matplotlib.ticker as mtick

from python.utils.utils import estimate_kde, plot_heatmap_scipy, return_segmentations_as_dataframe

def plot_element_heatmap(df : pd.DataFrame, xcol : str, ycol : str, filename : str, title : str, save_dir : str):

    """
    Function for plotting heatmap showing distribution of elements.

    df : pd.DataFrame
        segmentation / other data containing elements to be plotted.
    xcol : str
        name of xcol (coordinates) to be plotted
    ycol: str
        name of ycol (coordinates) to be plotted
    filename : str
        filename used in saving
    title : str
        Title to be plotted
    save_dir : str
        directory used to save the file
    """

    # Create plot
    fig, axs = plt.subplots(1,4, figsize=(11,10))

    # Plot
    for index, category in enumerate(["all categories", "web", "desktop", "mobile"]):
        data = df # Reset data
        ax = axs[index]

        if category != "all categories":
            data = data[data.category == category]

        ax = plot_heatmap_scipy(data, ax,  xcol= xcol, ycol = ycol, scatter=False, return_density_max=False, xmax=1, ymax=1, cmap="Spectral") # Plot

        # Set labels
        if index == 1:
            ax.set_xlabel("Horizontal element center\n(percentage of GUI)", size = 13, x = 1)
        if index == 0:
            ax.set_ylabel("Vertical element center\n(percentage of GUI)", size = 13)

        ax.set_title(f"{string.ascii_uppercase[index]}: {category.capitalize()}", fontweight="bold")
        ax.set_ylim(1,0)
        ax.set_xlim(0,1)
        ax.set_xticks([0,0.5, 1])
        ax.set_yticks([0,0.5, 1])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=315, size=6)
        ax.set_yticklabels(ax.get_xticklabels(), rotation=0, size=6)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    n = len(df)
    fig.suptitle(f"{title} N={n:,}", y=0.64, fontstyle = "italic")

    # Save
    plt.savefig(os.path.join(save_dir, f"{filename}.pdf"),dpi=300,bbox_inches="tight")

def main():
    """
    Plot Figs 7-8 for distributions of GUI elements.

    Saves plots to output/figs/supplement
    """
    # Set up directories
    segmentation_dir = os.path.join("data", "segmentation")
    vsgui10k_dir = os.path.join("data", "vsgui10k-images")
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Get visual search data
    search_times = pd.read_csv(os.path.join("output", "data", "vsgui10k_search_times.csv"))

    # Format segmentations from UEyes
    segmentation_dict = {k:[] for k in ["box_x_center", "box_y_center", "class", "ueyes_img_name"]} # Note the naming here, most references to ueyes_img_name throughout the code are the same as for img_name
    segmentation_df, _ = return_segmentations_as_dataframe(segmentation_dict = segmentation_dict, segmentation_dir=segmentation_dir, vsgui10k_dir=vsgui10k_dir)
    segmentation_df = segmentation_df.merge(search_times[["img_name", "category"]], right_on = "img_name", left_on="ueyes_img_name").drop_duplicates()

    # Plot UEyes (Fig. 8)
    plot_element_heatmap(segmentation_df, xcol = "box_x_center", ycol = "box_y_center", filename="supp_fig8", title="Text and image elements", save_dir=save_dir)

    # Format search times
    search_times["tgt_x_center"] = search_times.tgt_x + search_times.tgt_width/2
    search_times["tgt_y_center"] = search_times.tgt_y + search_times.tgt_height/2
    grouped = search_times[search_times.absent==False].groupby(["img_name", "category", "tgt_id"]).agg({"tgt_x_center" : "mean", "tgt_y_center" : "mean"}).reset_index()

    # Plot for VSGUI10k
    plot_element_heatmap(grouped, xcol = "tgt_x_center", ycol = "tgt_y_center", filename="supp_fig7", title="Target elements", save_dir=save_dir)


if __name__ == "__main__":
    main()