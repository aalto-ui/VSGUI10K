import pandas as pd
import os
import numpy as np
import logging

from python.utils.utils import get_string, add_segmentations, configure_logging

def main():
    """
    Pre-processes search times from Gazepoint fixation data.

    Saves search time file to output/data/vsgui10k_search_times.csv
    Saves logs to output/logs
    """

    # Save logs
    configure_logging(log_dir="output/logs", file_name="pre_process_search_times_logs")
    
    # Create folder for saving
    os.makedirs(os.path.join("output", "data"), exist_ok=True)

    # Get fixation info
    logging.info("Getting fixation file...")
    df_fixations = pd.read_csv(os.path.join("data", "vsgui10k_fixations.csv"))

    logging.info(f"Number of users in the fixation file: {len(np.unique(df_fixations.pid))}")
    logging.info(f"Number of images in the fixation file: {len(np.unique(df_fixations.new_img_name))}")
    logging.info(f"Number of trials in the fixation file: {len(np.unique(df_fixations.new_img_name)) / 4}")

    # Get directory where UEyes segmentations are stored --> needed for set size computations
    logging.info("Getting segmentations...")
    segmentation_dir = os.path.join("data", "segmentation")

    # Filter only search frames
    logging.info("Filtering to only search screens...")
    all_data = df_fixations[df_fixations.img_type == 2]
    logging.info(f"Number of trials: {len(np.unique(all_data.new_img_name))}")

    # Aggregate
    logging.info("Getting visual search times...")
    # Get visual search times as sum of fixation durations and trial times
    all_data_grouped = all_data.groupby(["pid", 
                                         "balanced_block_id", 
                                         "new_img_name", 
                                         "tgt_id"]).agg(
                                                        {"img_name" : lambda x : get_string(x),
                                                        "category" : lambda x : get_string(x),
                                                        "absent" : "mean",
                                                        "cue" : lambda x : get_string(x),  
                                                        "FPOGD" : "sum", # Aggregate to search times computed from fixation durations
                                                        "TIME" : "max", # Aggregate to search times computed from trial times
                                                        "FPOGID" : "count", # Aggregate to numbers of fixations
                                                        "tgt_text" : lambda x : get_string(x),
                                                        "tgt_color" : lambda x : get_string(x),
                                                        "tgt_location" : lambda x : get_string(x),
                                                        "tgt_x" : "mean",
                                                        "tgt_y" : "mean", 
                                                        "tgt_height" : "mean", 
                                                        "tgt_width" :"mean", 
                                                        "original_height" : "mean", 
                                                        "original_width" : "mean", 
                                                        "scale_x" : "mean",
                                                        "scale_y" : "mean",
                                                        }).reset_index()

    logging.info(f"Number of users in the fixation file: {len(np.unique(all_data_grouped.pid))}")
    
    # Rename columns
    all_data_grouped = all_data_grouped.rename(columns={"FPOGD" : "search_time_FPOGD", "TIME" : "search_time", "FPOGID" : "n_fixations"})
    logging.info(f"Number of trials: {len(np.unique(all_data_grouped.new_img_name))}")

    # Set size
    logging.info("Adding set sizes...")
    segmentation_dirs = [x[0] for x in os.walk(segmentation_dir) if "block" in x[0]]
    set_size_dict = add_segmentations(segmentation_dirs = segmentation_dirs, img_names = np.unique(df_fixations.img_name))
    df_set_size = pd.DataFrame(set_size_dict)

    # Add set sizes to the main data frame
    all_data_grouped = all_data_grouped.merge(df_set_size[["img_name", "set_size", "set_size_no_text"]], how="left", on="img_name")

    logging.info(f"Number of trials: {len(np.unique(all_data_grouped.new_img_name))}")

    logging.info(f"Number of users in the fixation file: {len(np.unique(all_data_grouped.pid))}")
    logging.info(f"Number of images in the fixation file: {len(np.unique(all_data_grouped.new_img_name))}")

    logging.info(f"Number of trials: {len(np.unique(all_data_grouped.new_img_name))}")

    # Make sure tgt_x and tgt_y are not defined for absent trials
    all_data_grouped.loc[all_data_grouped["absent"] == True, ["tgt_x"]] = np.nan
    all_data_grouped.loc[all_data_grouped["absent"] == True, ["tgt_y"]] = np.nan

    # Add target centers
    all_data_grouped["tgt_x_center"] = (all_data_grouped.tgt_x + (all_data_grouped.tgt_width / 2)) * all_data_grouped.scale_x
    all_data_grouped["tgt_y_center"] = (all_data_grouped.tgt_y + (all_data_grouped.tgt_height / 2)) * all_data_grouped.scale_y

    # Add aspect ratio
    all_data_grouped["aspect_ratio"] = (all_data_grouped.tgt_width * all_data_grouped.original_width*all_data_grouped.scale_x) / (all_data_grouped.tgt_height*all_data_grouped.original_height*all_data_grouped.scale_y)

    # Add target size
    all_data_grouped["tgt_size"] = (all_data_grouped.tgt_width * all_data_grouped.scale_x * all_data_grouped.tgt_height*all_data_grouped.scale_y)

    # Add target text length
    all_data_grouped["tgt_text_length"] = all_data_grouped.tgt_text.str.len()

    # Add normalised search times for each participant
    # logging.info("Normalising search times")
    # grouped_per_participant = all_data_grouped.groupby(["pid"]).agg({"search_time" : "max"}).reset_index()
    # grouped_per_participant = grouped_per_participant.rename(columns={"search_time": "search_time_max"})
    # logging.info(f"Number of participants: {len(np.unique(grouped_per_participant.pid))}")
    # all_data_grouped = all_data_grouped.merge(grouped_per_participant, on = "pid")
    # all_data_grouped["search_time_norm"] = all_data_grouped.search_time / all_data_grouped.search_time_max
    # logging.info(f"Number of trials: {len(np.unique(all_data_grouped.new_img_name))}")

    # logging.info("Adding distance to targets.")
    # # Add distance of target to fixation cross
    # all_data_grouped["tgt_x_center_from_fixation_cross"] = np.abs(all_data_grouped["tgt_x"] + all_data_grouped["tgt_width"] / 2 - 0.5) 
    # all_data_grouped["tgt_y_center_from_fixation_cross"] = np.abs(all_data_grouped["tgt_y"] + all_data_grouped["tgt_height"] / 2 - 0.5) 

    # Save visual search data
    logging.info("Saving data...")
    all_data_grouped.to_csv(os.path.join("output", "data", "vsgui10k_search_times.csv"), index=False)
    logging.info(f"Number of trials: {len(np.unique(all_data_grouped.new_img_name))}")
    logging.info("DONE")
if __name__=="__main__":
    main()