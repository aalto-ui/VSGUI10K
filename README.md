# VSGUI10K

This repository accompanies the paper "Understanding Visual Search in Graphical User Interfaces" published in International Journal of Human-Computer Studies (March, 2025).

## What's in this repository?

This repository contains scripts related to:

1. Generating the sample and stimuli for the controlled experiment (target, fixation cross, GUI and validation images).
2. Scripts for pre-processing data from Gazepoint.
3. Figures presented in the manuscript and supplement.
4. Fitted Linear Mixed-Effect Models and their variants presented in the paper.

(1-3) are in Python and (4) in R.

## Where's the data?

Please download the data from: https://osf.io/hmg9b/.

If you want to plot Fig. 10, you also need to download the UEyes data: https://zenodo.org/records/8010312.

## Getting started (Python)

Managing environments with ```conda```:

    conda env create -f environment.yml

    conda activate vsgui

## Quick start

There's a notebook that shows how to use the fixation data, in particular, displaying how to use the ```img_type```and ```FPOG``` variables correctly.

## If you want to replicate the experiment...

For the experiment, you have to 1) generate a sample, 2) balance the sample (=allocate 18 images per block) and 3) generate stimuli (target, fixation cross, GUI and validation image tuples). 

To generate sample:

    python3 -m python.src.experiment.00_generate_sample --target_filepath="vsgui10k_targets.csv" --fixation_cross

To balance said sample:

    python3 -m python.src.experiment.01_balance_sample --sample_filepath=sample/vsgui10k_sample.csv

To generate stimuli:

NOTE: Generating stimuli is the most time consuming step as 10,282*4 images need to be created. The script has an option to generate the stimuli for a set of blocks at a time (recommended). 

    python3 -m python.src.experiment.02_generate_stimuli --img_dir=data/vsgui10k-images --sample_filepath=output/sample/vsgui10k_sample_balanced.csv --target_filepath=data/vsgui10k_targets.csv --experiment_dir=output/experiment --font_file=<PATH TO FONT FILE> --max_block=1 --min_block=0 --balance --fixation_cross 

To generate stimuli images, note that you need to pass a font file in <PATH TO FONT .ttf FILE>.

## If you want to pre-process Gazepoint data

Then, search times are aggregated using:

    python3 -m python.src.pre_process.01_pre_process_search_times

Coverage data is calculated using (needed for Fig. 5, this takes a while):

    python3 -m python.src.pre_process.02_pre_process_coverage

Distance to targets is evaluated using (needed for Fig 12.):

    python3 -m python.src.pre_process.03_get_target_found_trials

The UEyes data is pulled from a directory <UEYES_DIR>; the path in your system is defined in <untracked_config/utnracked_config.py>. Process data using:

    python3 -m python.src.pre_process.ueyes.00_pre_process_ueyes

    python3 -m python.src.pre_process.ueyes.01_get_dims

## If you want to run the analyses...

Factor analysis and linear mixed effect regression are done in R. NOTE: AIM Metrics are calculated prior using: https://github.com/aalto-ui/aim

Scripts for factor analysis alongside instructions are documented in:

    fa.R

The LMER (and GLMER) models are presented in:

    1-controlled.R
    2-visual-complexity.R
    3-interaction.R

## If you want to plot the figures...

Make sure that you have run above (except stimuli generation scripts) before plotting. The scripts used to produce figures are stored in <python/plots>. Run each script to produce the desired plot. For instance,

    python3 -m python.plots.fig0_graphical_abstract

Note that regression-related Figures (3, 7, 9 in main paper) are plotted via R.

## Authors and acknowledgment
Authors of the accompanying publication contributed to this project. Generative AI was used in preparing this code (Aalto AI, GPT4o, February--March 2025).

## Citation
If you use ```VSGUI10K``` in your own work, please cite: 

    @article{putkonen2025_vsgui10k,
            title = {Understanding visual search in graphical user interfaces},
            journal = {International Journal of Human-Computer Studies},
            volume = {199},
            pages = {103483},
            year = {2025},
            issn = {1071-5819},
            doi = {https://doi.org/10.1016/j.ijhcs.2025.103483},
            author = {Aini Putkonen and Yue Jiang and Jingchun Zeng and Olli Tammilehto and Jussi P.P. Jokinen and Antti Oulasvirta},
            }
