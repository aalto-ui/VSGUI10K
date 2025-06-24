import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    # Define directories for saving the figure
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Read search times data from a CSV file
    search_times = pd.read_csv(os.path.join("output", "data", "vsgui10k_search_times.csv"))

    # Create subplots 
    fig, axs = plt.subplots(1, 2, figsize=(8, 2))

    ax0 = axs[0]
    ax1 = axs[1]

    # Calculate actual image width and height based on scaling factors (screenshot sizes were adjusted for the experiment)
    search_times["img_width"] = search_times.scale_x * 1920
    search_times["img_height"] = search_times.scale_y * 1920

    # Plot histogram of image width categorized by UI types
    sns.histplot(
        data=search_times,
        x="img_width",
        ax=ax0,
        hue="category",
        element="step",
        legend=False
    )

    # Plot histogram of image height categorized by UI types
    sns.histplot(
        data=search_times,
        x="img_height",
        ax=ax1,
        hue="category",
        element="step"
    )

    # Set x-axis labels and remove y-axis label for the second plot
    ax0.set_xlabel("Image width (px)")
    ax1.set_xlabel("Image height (px)")
    ax1.set_ylabel("")

    # Move the legend for the second subplot to the upper left position
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(-0.7, 1.3), ncol=3, title=None)

    # Set y-axis limits for both subplots
    ax0.set_ylim(0, 4000)
    ax1.set_ylim(0, 4000)

    # Set x-axis limits for both subplots
    ax0.set_xlim(0, 2000)
    ax1.set_xlim(0, 2000)

    # Save the figure as a PDF file in the specified directory
    fig.savefig(os.path.join(save_dir, "supp_fig2.pdf"), dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()