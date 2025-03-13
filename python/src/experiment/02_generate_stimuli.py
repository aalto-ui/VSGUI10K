import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
import argparse
from tqdm import tqdm

from python.utils.utils import map_nans

parser = argparse.ArgumentParser()

parser.add_argument("--img_dir", help="Directory name where images are retrieved from.", type=str, required=True)
parser.add_argument("--sample_filepath", help="Directory name where sample is retrieved from.", type=str, required=True)
parser.add_argument("--target_filepath", help="Directory name where target images are retrieved from.", type=str, required=True)
parser.add_argument("--experiment_dir", help="Directory where experiment images are saved.", type=str, required=True)
parser.add_argument("--max_block", help="Maximum block index to generate.", type=int, default=10000)
parser.add_argument("--min_block", help="Minimum block index to generate.", type=int, default=0)
parser.add_argument("--font_file", help="Path to font.", type=str, required=True)
parser.add_argument("--balance", help="whether to run balanced sample", action = "store_true")
parser.add_argument("--fixation_cross", help="whether to force fixation cross", action = "store_true")

def main():
 

    args = parser.parse_args()

    balance = args.balance # Whether to use balanced sample or not (impacts number of images per block)

    ##################################
    ## Gather directories and files ##
    ##################################

    SAMPLE_FILEPATH = os.path.join(args.sample_filepath)
    TARGET_FILEPATH = os.path.join(args.target_filepath)
    IMG_DIR = os.path.join(args.img_dir)
    EXP_DIR = os.path.join(args.experiment_dir)

    os.makedirs(EXP_DIR, exist_ok=True)

    df_sample = pd.read_csv(SAMPLE_FILEPATH, na_filter=False)
    df_target = pd.read_csv(TARGET_FILEPATH, na_filter=False)

    #################
    ## FILTER DATA ##
    #################

    df_sample = df_sample[(df_sample.block_id >= args.min_block) & (df_sample.block_id <= args.max_block)].reset_index(drop=True) # How many blocks to create

    map_nans(df_sample)

    df_sample = df_sample.dropna(subset = ["tgt_id"], axis='rows').reset_index(drop=True)

    idx = df_sample.index[(df_sample.absent == True) & (df_sample.absent_tgt_id.isna())]

    df_sample = df_sample.drop(idx).reset_index(drop=True)

    ########################
    ## SET FIXATION CROSS ##
    ########################

    if not args.fixation_cross:
        if int(df_sample.new_img_name[1][:-4])-int(df_sample.new_img_name[0][:-4]) == 4:
            fixation_cross = True
        else:
            fixation_cross = False
    else:
        fixation_cross = args.fixation_cross

    stimuli_dict = {k : [] for k in ["img_name", "x_upper_left", "y_upper_left", "screenshot_height", "screenshot_width"]}

    #####################################
    ## Generate images for each sample ##
    #####################################

    for sample in tqdm(df_sample.itertuples()):
        
        if balance: # Whether to use the stored balanced_block_id
            block_dir = os.path.join(EXP_DIR, f"block_{sample.balanced_block_id}")
        else:
            block_dir = os.path.join(EXP_DIR, f"block_{sample.block_id}")

        # Set filenames (3 images if no fixation cross, otherwise 4; target, (cross), visual search, validation)
        os.makedirs(block_dir, exist_ok=True)
        tgt_filename = sample.new_img_name
        stimuli_dict["img_name"].append(tgt_filename)

        if fixation_cross:
            cross_filename = f"{int(tgt_filename[:-4]) + 1}.jpg"
            screenshot_filename = f"{int(tgt_filename[:-4]) + 2}.jpg"
            validation_filename = f"{int(tgt_filename[:-4]) + 3}.jpg"

        else:
            screenshot_filename = f"{int(tgt_filename[:-4]) + 1}.jpg"
            validation_filename = f"{int(tgt_filename[:-4]) + 2}.jpg"
        
        # Retrieve target from another image if absent
        if sample.absent:
            tgt_info = df_target[(df_target.ueyes_img_name == sample.absent_img) & (df_target.tgt_id == sample.absent_tgt_id)]
            try:
                absent_img_filepath = os.path.join(IMG_DIR, sample.absent_img)
                absent_screenshot = Image.open(absent_img_filepath) 
            except Exception as e:
                print (e)
                absent_img_filepath = os.path.join(IMG_DIR, sample.absent_img.replace("_", " "))
                absent_screenshot = Image.open(absent_img_filepath) 
        else:
            tgt_info = df_target[(df_target.ueyes_img_name == sample.img_name) & (df_target.tgt_id == sample.tgt_id)]
        
        # Resolve some issues with whitespaces in original filenames
        try:
            img_filepath = os.path.join(IMG_DIR, sample.img_name)
            screenshot = Image.open(img_filepath)
        except Exception as e:
            print (e)
            img_filepath = os.path.join(IMG_DIR, sample.img_name.replace("_", " "))
            screenshot = Image.open(img_filepath)

        # Set image size
        screenshot_width, screenshot_height = screenshot.size
        stimuli_dict["screenshot_width"].append(screenshot_width)
        stimuli_dict["screenshot_height"].append(screenshot_height)

        # Generate canvas
        width, height = 1920, 1200
        aspect_ratio = width / height

        # Generate cross image
        if fixation_cross:
            cross_img = Image.new(mode="RGB", size=(width, height), color=(255,255,255))
            draw = ImageDraw.Draw(cross_img)
            text = "+"
            font = ImageFont.truetype(args.font_file, 60)
            _, _, marker_width, marker_height = font.getbbox(f"{text}")
            draw.text((width/2 - marker_width / 2, height /2 - marker_height / 2),text,(0,0,0),font=font)
            cross_img.save(os.path.join(block_dir, cross_filename))

        # Set size of the image
        if sample.category == "web":
            img_height = int(screenshot_height / 0.75) # Max 0.75 of screen height
        elif sample.category == "mobile": 
            img_height = int(screenshot_height / 0.61) # Scaled to natural viewing distance
        elif sample.category == "desktop": # Max 0.75 of screen height
            if screenshot_height < height * 0.75:
                img_height = int(screenshot_height / (screenshot_height /height))
            else: 
                img_height = int(screenshot_height / 0.75)

        img_width = int(img_height * aspect_ratio)
        size = (img_width, img_height)
        img = Image.new(mode="RGB", size=size, color=(255,255,255))

        screenshot_img = img.copy()

        # Get coordinates of the screenshot upper-left
        x, y = int(img_width / 2) - int(screenshot_width / 2), int(img_height / 2) - int(screenshot_height / 2)
        stimuli_dict["x_upper_left"].append(x / img_width)
        stimuli_dict["y_upper_left"].append(y / img_height)

        # Add screenshot
        screenshot_img.paste(screenshot, box = (x,y, x + screenshot_width, y+screenshot_height))
        draw = ImageDraw.Draw(screenshot_img)
        x, y = int(img_width / 2) - int(screenshot_width / 2), int(img_height / 2) - int(screenshot_height / 2)
        shape = [(x, y), (x + screenshot_width, y+screenshot_height)]
        draw.rectangle(shape, outline ="red", width=3)
        
        # Save screenshot image
        screenshot_img = screenshot_img.resize((width, height))
        screenshot_img.save(os.path.join(block_dir, screenshot_filename))

        # Generate target screen
        tgt_img = img.copy()

        # Generate correct cue
        if sample.text_description == "i": # Image cue
        
            original_width, original_height = tgt_info.original_width.item(), tgt_info.original_height.item()
            tgt_x,tgt_y = int(tgt_info.tgt_x.item()/100*original_width), int(tgt_info.tgt_y.item()/100*original_height)
            tgt_height, tgt_width = int(tgt_info.tgt_height.item()/100*original_height), int(tgt_info.tgt_width.item()/100*original_width)

            box = tgt_x, tgt_y, tgt_x + tgt_width, tgt_y+tgt_height

            if sample.absent:
                screenshot = absent_screenshot
            
            tgt = screenshot.crop(box)

            x, y = int(img_width / 2) - int(tgt_width / 2), int(img_height / 2) - int(tgt_height / 2)
            tgt_img.paste(tgt, box = (x,y, x + tgt_width, y+tgt_height))

            tgt_img = tgt_img.resize((width, height))
        
        else: # Text and Text + color cues
            tgt_img = tgt_img.resize((width, height))
            draw = ImageDraw.Draw(tgt_img)
            font = ImageFont.truetype(args.font_file, 30)

            if sample.text_description == "t": # Text
                text = f'Text: "{tgt_info.tgt_text.item()}"'
                _, _, marker_width, marker_height = font.getbbox(f"{text}")
                draw.text((width/2 - marker_width / 2, 600 - marker_height / 2),text,(0,0,0),font=font)
            if sample.text_description == "tc": # Color
                text = f'Text: "{tgt_info.tgt_text.item()}"'
                _, _, marker_width, marker_height = font.getbbox(f"{text}")
                draw.text((width/2 - marker_width / 2, 600 - marker_height / 2),text,(0,0,0),font=font)
                text = f'Background color: {tgt_info.tgt_color.item()}'
                _, _, marker_width, marker_height = font.getbbox(f"{text}")
                draw.text((width/2 - marker_width / 2, 650 - marker_height / 2),text,(0,0,0),font=font)


            text = "Find a target with the following features"
            font = ImageFont.truetype(args.font_file, 40)
            _, _, marker_width, marker_height = font.getbbox(f"{text}")
            draw.text((width/2 - marker_width / 2, 500 - marker_height / 2),text,(0,0,0),font=font)

        # Set instructions
        draw = ImageDraw.Draw(tgt_img)
        font = ImageFont.truetype(args.font_file, 50)
        text = "Next target"
        _, _, marker_width, marker_height = font.getbbox(f"{text}")
        draw.text((width/2 - marker_width / 2,50),text,(255,0,0),font=font)

        text = "Press spacebar when ready"

        _, _ ,marker_width, marker_height = font.getbbox(f"{text}")
        draw.text((width/2 - marker_width / 2, 1050),text,(255,0,0),font=font)

        # Save target image
        tgt_img.save(os.path.join(block_dir, tgt_filename))

        # Generate validation image
        click_img = img.copy()
        rectangle_img = ImageDraw.Draw(click_img)  
        x, y = int(img_width / 2) - int(screenshot_width / 2), int(img_height / 2) - int(screenshot_height / 2)
        shape = [(x, y), (x + screenshot_width, y+screenshot_height)]
        rectangle_img.rectangle(shape, outline ="red", width=3)

        click_img = click_img.resize((width, height))

        draw = ImageDraw.Draw(click_img)  
        font = ImageFont.truetype(args.font_file, 40)

        text = "Look at the location of the target for 3 seconds. If the target was absent, look at the red square."
        
        _, _, marker_width, marker_height = font.getbbox(f"{text}")
        draw.text((width/2 - marker_width / 2,50),text,(255,0,0),font=font)

        # Create a rectangle for target absent
        text = "Target absent"
        font = ImageFont.truetype(args.font_file, 20)
        target_park = [(width - 200, height - 100), (width-50, height - 50)]
        draw.rectangle(target_park, outline ="red", width=3, fill = "red")
        draw.text((width - 185, height- 85 ),text,(0,0,0),font=font)

        click_img.save(os.path.join(block_dir, validation_filename))

    # Optional saving of stimuli info
    #df_stimuli = pd.DataFrame(stimuli_dict)
    #df_stimuli.to_csv(os.path.join("data", "stimuli_info.csv"))

if __name__ == "__main__":
    main()