import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import string

def main():
    # Define directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Read the sample data from a CSV file
    df_sample = pd.read_csv(os.path.join("output", "sample", "vsgui10k_sample_balanced.csv"))

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    palette = ["#5BA300", "#ccff66", "#0073E6"]
    hue_order = ['mobile', 'desktop', 'web']
    xlabels = ["Target present", "Target absent"]
    
    # Iterate through 'False' and 'True' for target absent and present conditions
    for index, absent in enumerate([False, True]):
        ax = axs[index]
        
        # Create count plot based on the presence/absence of the target
        sns.countplot(
            data=df_sample[df_sample.absent == absent],
            x="text_description",
            ax=ax,
            hue="category",
            palette=palette,
            hue_order=hue_order,
            order=["i", "t", "tc"] # Set order for cues
        )
        ax.set_ylim(0, 1900)  # Set y-axis limit

        # Set y-axis labels: 'Count' for the first subplot, no label for the second
        if index == 0:
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("")
            ax.set_yticks([])  # Remove y-axis ticks for the second subplot
        
        # Set x-axis labels with bold font weight
        ax.set_xlabel(xlabels[index], fontweight="bold")
        
        # Add bar labels to the count plot containers
        for container in ax.containers:
            ax.bar_label(container)
        
        # Remove legend from first subplot, add legend to the second subplot
        if index == 0:
            ax.get_legend().remove()
        else:
            ax.legend(labels=['Mobile UI', 'Desktop UI', "Webpage"])
        
        # Set custom x-axis tick labels
        ax.set_xticklabels(["Image", "Text", "Text+color"])
        
        # Annotate subplot with the appropriate letter
        ax.text(0, 1.08, string.ascii_uppercase[index], transform=ax.transAxes,
                size=15, weight='bold')
    
    # Move the legend for the second subplot to the lower center position
    sns.move_legend(axs[1], "lower center", bbox_to_anchor=(-0.1, -0.6), ncol=3)

    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    # Save the figure as a PDF file in the specified directory
    plt.savefig(os.path.join(save_dir, "supp_fig1.pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()