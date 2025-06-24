import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import matplotlib as mpl

from python.utils.utils import get_string, map_color_strings

def main():
    # Define directories for saving
    save_dir = os.path.join("output", "figs", "supplement")
    os.makedirs(save_dir, exist_ok=True)

    # Read the sample data and target data from CSV files into dataframes
    df_sample = pd.read_csv(os.path.join("output", "sample", "vsgui10k_sample_balanced.csv"))
    df_target = pd.read_csv(os.path.join("data", "vsgui10k_targets.csv"))

    # Define the aggregation dictionary to summarize the target data
    agg_dict = {
        "index": "count",
        "tgt_x": "mean", 
        "tgt_y": "mean",
        "tgt_width": "mean",
        "tgt_height": "mean",
        "original_width": "mean",
        "original_height": "mean",
        "absent": "mean",
        "category_x": lambda x: get_string(x),
        "tgt_color": lambda x: get_string(x),
        "tgt_text": lambda x: get_string(x)
    }

    # Merge the sample and target data 
    filtered = pd.merge(df_target, df_sample, how="inner", on=["tgt_id"]).reset_index()
    
    # Group the filtered data by 'tgt_id' and apply the aggregation functions
    grouped_tgt = filtered.groupby("tgt_id").agg(agg_dict).reset_index()

    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))
    mpl.rc('font', family="Arial")

    # Plot the target width distribution
    ax = axs[0, 0]
    ax.set_ylabel("Count", size=15, y=-0.5)
    grouped_tgt["true_width"] = grouped_tgt.tgt_width / 100 * grouped_tgt.original_width
    sns.histplot(data=grouped_tgt, x="true_width", ax=ax, bins=25)
    ax.set_xlabel("Target width (px)", size=13)
    ax.text(-0.15, 1.1, string.ascii_uppercase[0], transform=ax.transAxes, size=15, weight='bold')
    ax.set_ylim(0, 1000)
    ax.set_xlim(0, 1300)

    # Plot the target height distribution
    ax = axs[0, 1]
    grouped_tgt["true_height"] = grouped_tgt.tgt_height / 100 * grouped_tgt.original_height
    sns.histplot(data=grouped_tgt, x="true_height", ax=ax, bins=25)
    ax.set_xlabel("Target height (px)", size=13)
    ax.set_ylabel("")
    ax.text(-0.15, 1.1, string.ascii_uppercase[1], transform=ax.transAxes, size=15, weight='bold')
    ax.set_ylim(0, 1000)
    ax.set_xlim(0, 1300)

    # Plot the target background colors distribution
    ax = axs[1, 0]
    grouped_tgt["tgt_color"] = grouped_tgt["tgt_color"].apply(lambda x: map_color_strings(x))
    grouped_tgt = grouped_tgt[(grouped_tgt.tgt_color != "multicolot") & (grouped_tgt.tgt_color != "button")]
    sns.countplot(data=grouped_tgt, x="tgt_color", edgecolor="black", ax=ax, palette=[
        'grey', 'white', 'black', "blue", "red", "yellow", "orange", "maroon", "purple", "green", "pink", "brown"
    ])
    ax.set_xlabel("Target background color", size=13)
    xlabels = ax.get_xticklabels()
    ax.set_xticklabels(xlabels, rotation=50)
    ax.text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax.transAxes, size=15, weight='bold')
    ax.set_ylabel("")

    # Plot the distribution of text length in the target text
    ax = axs[1, 1]
    grouped_tgt["text_length"] = grouped_tgt["tgt_text"].str.len()
    sns.histplot(data=grouped_tgt, x="text_length", ax=ax, bins=25)
    ax.set_xlabel("# characters in the text", size=13)
    ax.set_ylabel("")
    ax.text(-0.15, 1.1, string.ascii_uppercase[3], transform=ax.transAxes, size=15, weight='bold')
    ax.set_xlim(0, 50)

    # Adjust the layout and save the figure to a PDF file
    fig.align_labels()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    fig.savefig(os.path.join(save_dir, "supp_fig6.pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()