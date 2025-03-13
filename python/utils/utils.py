import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict
import json
import os
from tqdm import tqdm
import shapely
from shapely.ops import unary_union
import itertools
import logging
import string
import matplotlib.lines as lines

def configure_logging(file_name: str, log_dir: str):

    '''
    Configure logging for search time generation.

    Args:
    -----
        file_name : name of the file where logs will be saved
        dir : directory where logs will be saved
    '''

    log_folder = os.path.join(log_dir, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, file_name), level=logging.INFO, filemode="w", format='%(asctime)s %(levelname)-8s %(message)s')

def add_segmentations(segmentation_dirs : list, img_names : np.array) -> Dict:
    """
    Add segmentations from UEyes for set size analysis.

    Args:
    -----
    segmentations_dirs : list
        List of directories where UEyes segmentations are stored.
    img_names : np.array
        Images for which set sizes are calculated.

    Returns:
    --------
    set_size_dict : Dict
        Dictionary containing set sizes.
    """
    
    # Initalise set size dictionary
    set_size_dict = {k : [] for k in ["img_name", "set_size", "set_size_no_text"]}
    
    # Initalise buffer for json files
    all_json_files =[]

    # Gather json files
    for directory in segmentation_dirs:
        jsons = [name for name in os.listdir(directory) if "json" in name]
        for json in jsons:
            all_json_files.append(os.path.join(directory, json))
    
    # Obtain set size for each image from segmentations
    for img_name in img_names:
        
        json_file = [f for f in all_json_files if img_name[:-4] in f][0]
        segmentation = get_segmentation(json_file)

        segmentation_df = pd.DataFrame(segmentation)
        # Filter out Background and Face classes
        segmentation_filtered = segmentation_df[(segmentation_df["class"]!="Background") & (segmentation_df["class"]!="Face")]
        # Filter out Background, Face and Text classes
        segmentation_filtered_no_text = segmentation_df[(segmentation_df["class"]!="Background") & (segmentation_df["class"]!="Face")& (segmentation_df["class"]!="Text")]

        # Store results
        set_size_dict["img_name"].append(img_name)
        set_size_dict["set_size"].append(len(segmentation_filtered))
        set_size_dict["set_size_no_text"].append(len(segmentation_filtered_no_text))

    return set_size_dict

def map_nans(df : pd.DataFrame) -> None: 
    """
    Replace missing values within a DataFrame with `numpy.nan` (in place).

    Args:
    -----
    df : pd.DataFrame
        The DataFrame in which to replace and fill missing values.
    """

    df.replace('None', np.nan, inplace=True)
    df.replace('np.nan', np.nan, inplace=True)
    df.replace('NaN', np.nan, inplace=True)
    df.replace('', np.nan, inplace=True)
    df.replace('nan', np.nan, inplace=True)

    df.fillna(value=np.nan, inplace = True)

def get_string(col):
    """
    Map dataframe column to a string.

    Args:
    -----
    col : pd.Series
        Column to be mapped
    """
    arr = np.asarray(col)
    
    if np.all(arr == arr[0]):
        return arr[0]


def draw_scanpath(xs : np.array, ys : np.array, ts:np.array, img_path : str, ax, color = [54/255, 100/255, 139/255], draw=True):
    """
    Draw scanpaths. Check: https://github.com/YueJiang-nj/UEyes-CHI2023/blob/main/data_processing/aggregate_data.py

    Args:
    -----
    xs :  [np.array, list]  
        x-coordinates of fixation points.
    ys : [np.array, list]  
        y-coordinates of fixation points.
    ts : [np.array, list] 
        Fixation durations.
    img_path : str
        Path to image file.
    ax : matplotlib.axes.Axes
        Axis object for plotting
    color : list
        Colors used in plotting
    draw : bool
        If true, draw the scanpath

    Returns:
    --------
    _img : PIL Image
        Image object (where scanpath is drawn)
    """

    colors = [[100, 180, 255, 255],
              [50, 50, 50, 255]]

    cm = LinearSegmentedColormap.from_list('', np.array(colors) / 255, 256)    

    _img = Image.open(img_path).convert("RGB")
    _img.putalpha(int(255 * 0.6))
    img = np.array(_img)
    
    xs = xs*img.shape[1]
    ys = ys*img.shape[0]

    width = img.shape[1]
    height = img.shape[0]
    w = 1
    h = 1
    plt.gray()
    
    ax.imshow(img)
    cmap = (cm(np.linspace(0, 1, 2 * len(xs) - 1)) * 255).astype(np.uint8)

    if draw:
        for i in range(len(xs)):
            if i > 0:
                ax.axes.arrow(
                    xs[i - 1]*w,
                    ys[i - 1]*h,
                    xs[i]*w - xs[i - 1]*w,
                    ys[i]*h - ys[i - 1]*h,
                    width=min(width, height) / 300 * 2,
                    color=color,
                        alpha=1,)

        for i in range(len(xs)):
            if i == 0:
                edgecolor = 'red'
            else:
                edgecolor = "black"
            circle = plt.Circle(
                (xs[i]*w, ys[i]*h),
                radius=min(width, height) / 35 * ts[i] * 2,
                edgecolor=edgecolor,
                facecolor=cmap[i * 2] / 255.,
                linewidth=1
                )
            ax.axes.add_patch(circle)
        
    return _img

def draw_target(img_name : str, tgt_id : str, img : Image, df : pd.DataFrame, ax, scale=True):
    """
    Draw target on an image.

    Args:
    -----
    img_name : str 
        Name of the image.
    tgt_ id : str
        Unique target identifier.
    img : PIL Image
        Image where target is drawn.
    df : pd.DataFrame
        Target information dataframe.
    ax : matplotlib.axes.Axes
        Axis object for drawing.
    scale : bool
        Whether to scale the target coordinates or not.

    Returns:
    --------
    img : PIL Image
        Image with target drawn.
    """
    
    try:
        img = Image.fromarray(img)
    except:
        img = img

    df_tmp = df[(df.ueyes_img_name == img_name) & (df.tgt_id == tgt_id)] # Note that target file has both img_name and ueyes_img_name
    tgt_x = df_tmp.tgt_x.item() / 100
    tgt_y = df_tmp.tgt_y.item() / 100
    tgt_height = df_tmp.tgt_height.item() / 100
    tgt_width = df_tmp.tgt_width.item() / 100
    original_width = df_tmp.original_width.item()
    original_height = df_tmp.original_height.item()
    category = df_tmp.category.item()

    if scale:
        # Whether to scale original target coordinates, depends on whether target drawn on screenshot or monitor dimensions
        scale, scale_x, base_height, base_width = get_screenshot_dims(img_name, tgt_id, img, df)

        _x = base_width * (1-scale_x) / 2 + (base_width * scale_x) * tgt_x
        _y = base_height * (1-scale) / 2 + (base_height * scale) * tgt_y
        
        _height = (tgt_height) * (scale * base_height)
        _width = (tgt_width) * (scale_x * base_width)

    else:
        _x = tgt_x * original_width
        _y = tgt_y * original_height
        _height = tgt_height * original_height
        _width = tgt_width * original_width

    draw = ImageDraw.Draw(img)
    draw.rectangle([_x, _y, _x+_width, _y+_height], outline=(0,255,0), width=5)
    kwargs = {"horizontalalignment" : "center"}
    plt.gray()
    ax.annotate('Target', xy=(_x, _y), xytext=(-0.4, 0.5), xycoords = 'data', textcoords = "axes fraction", arrowprops=dict(facecolor='white', shrink=0.05), **kwargs)
    ax.scatter(_x+_width/2, _y+_height/2, marker="x", color="red")
    ax.imshow(img)

    return img

def plot_heatmap_scipy(data: pd.DataFrame, ax, scatter = True, return_density_max=False, xcol = "FPOGX", ycol = "FPOGY", xmin = 0, xmax = 1, ymin = 0, ymax = 1, cmap="bwr"):

    """
    Plots a heatmap based on the Kernel Density Estimate (KDE) of the specified columns of a DataFrame onto a given matplotlib axis.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html and https://github.com/scipy/scipy/blob/main/LICENSE.txt

    Args:
    -----
    data : pd.DataFrame
        The input DataFrame containing eye-tracking data.
    ax : matplotlib.axes.Axes
        The matplotlib axis on which to plot the heatmap.
    scatter : bool, optional
        Whether to overlay scatter points of the original data on the heatmap. Default is True.
    return_density_max : bool, optional
        Whether to mark the point with highest density with an 'X'. Default is False.
    xcol : str, optional
        The name of the column for the x-axis data. Default is "FPOGX".
    ycol : str, optional
        The name of the column for the y-axis data. Default is "FPOGY".
    xmin : float, optional
        The minimum value of the x-axis for the KDE grid. Default is 0.
    xmax : float, optional
        The maximum value of the x-axis for the KDE grid. Default is 1.
    ymin : float, optional
        The minimum value of the y-axis for the KDE grid. Default is 0.
    ymax : float, optional
        The maximum value of the y-axis for the KDE grid. Default is 1.
    cmap : str, optional
        The colormap to be used for the heatmap. Default is "bwr".

    Returns:
    --------
    matplotlib.axes.Axes
        The axis with the plotted heatmap (and optionally scatter points, and maximum density "X").
    """

    # Estimate KDE and plot
    z, xmin, xmax, ymin, ymax, x_density_max, y_density_max = estimate_kde(data, xcol, ycol, xmin, xmax, ymin, ymax)
    ax.imshow(np.rot90(z), cmap=cmap,
          extent=[xmin, xmax, ymin, ymax], alpha=0.7)

    # Scatter fixations on the plot
    if scatter:
        ax.plot(data[xcol], data[ycol],"k.", markersize=2)
     
    # Invert axis as in Gazepoint (0,0) is upper-left
    ax.invert_yaxis()

    # Draw "X" for highest density 
    if return_density_max:
        ax.scatter(x_density_max, y_density_max, s=100, c='b', marker = 'X',  zorder=10)
    
    return ax

def get_screenshot_dims(img_name : str, tgt_id : str, img : Image, df : pd.DataFrame):
    """
    Get dimensions of a screenshot on the screen it was shown on in the user study.

    Args:
    -----
    img_name : str
        Name of the image of interest.
    tgt_id : str
        Target ID.
    img : PIL Image
        Image object
    df : pd.DataFrame
        Dataframe where category information is obtained.

    Returns:
    --------
    scale : float
        Height of the image as percentage of the monitor viewport.
    scale_x : float
        Width of the image as percentage of the monitor viewport.
    base_height : float
        Original height of the image
    base_width : float
        Original width of the image
    """
   
    # Establish base width
    base_width = img.size[0]
    base_height = img.size[1]
    
    # Get aspect ratio
    aspect_ratio_base = base_width / base_height

    # Get some other information about the image
    df_tmp = df[(df.ueyes_img_name == img_name) & (df.tgt_id == tgt_id)]
    original_width = df_tmp.original_width.item()
    original_height = df_tmp.original_height.item()
    category = df_tmp.category.item()
    
    aspect_ratio = original_width / original_height
    
    # Get height of the image as percentage of screen, note that this is also used when creating the stimuli for the experiment
    if category == "web":
        scale = 0.75
        img_height = original_height / scale
    elif category == "mobile": 
        scale = 0.61
        img_height = original_height /  scale
    elif category == "desktop": 
        if original_height < base_height * 0.75:
            scale = original_height / base_height
            img_height = original_height / scale
        else: 
            scale = 0.75
            img_height = original_height / 0.75

    img_width = img_height * aspect_ratio_base
    
    scaled_img_width = (original_height) * aspect_ratio
    
    # Get width of the image as percentage of screen
    scale_x = scaled_img_width / img_width

    return scale, scale_x, base_height, base_width 

def map_color_strings(string):

    """
    Maps color names to consistent representations. This is done as there are some typos in the annotations.

    Args:
    -----
    string : str
        The input string representing a color.

    Returns:
    --------
    str: A standardized color name based on the input string.
    """
    
    if string == "White" or string == "button" or string == "whitw" or string == "whtie" or "white" in string:
        string = "white"
        
    if string == "Blue" or string == "light blue" or string == "aquamarine" or string == "teal" or string == "grey blue" or string == "dark blue" or "blue" in string:
        string = "blue"
    
    if string == "Grey" or string == "gray" or string == "gery" or string == "light grey" or "grey" in string:
        string = "grey"
        
    if string == "light brown" or string == "terracotta" or "brown" in string:
        string = "brown"
        
    if string == "light yellow" or string == "Yellow" or "yellow" in string:
        string = "yellow"
        
    if string == "balck":
        string = "black"
    
    if string == "oragnge" or string == "Orange" or "orange" in string:
        string = "orange"
    
    if "red" in string:
        string = "red"
        
    if "multicolott" == string or "multi color" == string or "mulmticolor" == string or "muti color" == string or "multicolott" == string or "colot" in string or "multicolor" == string:
        string = "multi"
        
    if "green " == string or "green" in string:
        string = "green"
        
    return string

def estimate_kde(data: pd.DataFrame, xcol = "FPOGX", ycol = "FPOGY", xmin = 0, xmax = 1, ymin = 0, ymax = 1):
    
    """
    Estimates the Kernel Density Estimate (KDE) for the specified columns of a DataFrame and identifies the (x, y) coordinates with the highest density.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html and https://github.com/scipy/scipy/blob/main/LICENSE.txt

    Args:
    -----
    data : pd.DataFrame
        The input DataFrame containing the data.
    xcol : str
        The name of the column for the x-axis data. Default is "FPOGX".
    ycol : str 
        The name of the column for the y-axis data. Default is "FPOGY".
    xmin : float 
        The minimum value of the x-axis for the KDE grid. Default is 0.
    xmax :float 
        The maximum value of the x-axis for the KDE grid. Default is 1.
    ymin : float 
        The minimum value of the y-axis for the KDE grid. Default is 0.
    ymax :float
        The maximum value of the y-axis for the KDE grid. Default is 1.

    Returns:
    --------
    z : np.ndarray
        The estimated density values on the KDE grid.
    xmin : float 
        The minimum x value of the KDE grid.
    xmax : float 
        The maximum x value of the KDE grid.
    ymin : float
        The minimum y value of the KDE grid.
    ymax :float
        The maximum y value of the KDE grid.
    x_density_max : np.ndarray
        The x-coordinate(s) with the highest density.
    y_density_max : np.ndarray
        The y-coordinate(s) with the highest density.
    """

    # Creates a grid of x and y values over the specified range.
    x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([data[xcol]*xmax, data[ycol]*ymax])

    # Create the kernel
    kernel = stats.gaussian_kde(values)
    z = np.reshape(kernel(positions).T, x.shape)
    max_idx = np.where(kernel.evaluate(values) == kernel.evaluate(values).max())

    # Get the point where the density is highest
    x_density_max = data[xcol].to_numpy()[max_idx]*xmax
    y_density_max = data[ycol].to_numpy()[max_idx]*ymax

    # Test that the same (x,y) coordinates are returned (no multiple maxima)
    test_x = np.all(x_density_max == x_density_max[0])
    test_y = np.all(y_density_max == y_density_max[0])

    return z, xmin, xmax, ymin, ymax, x_density_max, y_density_max

def get_segmentation(json_file, return_img=False):

    """
    Return segmentations from json files.

    Args:
    -----
    json_file : str
        Name of json file
    return_img : bool
        Whether to return images or not

    Returns:
    --------
    segmentation["compos"] : list
        Segmentation retrieved from the json file.
    """

    # Open JSON file
    with open(json_file) as f:
         segmentation = json.load(f)
    f.close()
    
    # Return segmentation
    if return_img:
        return segmentation["compos"], segmentation["img"]
    else:
        return segmentation["compos"]

def return_segmentations_as_dataframe(segmentation_dict: Dict, segmentation_dir : str, vsgui10k_dir : str):
    """
    Return segmentations as a dataframe.

    Args:
    -----
    segmentation_dict : Dict
        Dictionary with required columns
    segmentation_dir : str
        Directory where segmentations are stored
    vsgui10k_dir : str
        Directory where vsgui10k images are stored
    
    Returns:
    --------
    segmentation_df : pd.DataFrame
        Segmentations as a dataframe
    failed_images : list
        Any images where retrieving segmentations failed
    """

    # Get names of folders where segmentations are saved
    segmentation_dirs = [x[0] for x in os.walk(segmentation_dir) if "block" in x[0]]
    
    # Get names of all json files
    all_json_files =[] 
    for directory in segmentation_dirs:
        jsons = [name for name in os.listdir(directory) if "json" in name]
        for json in jsons:
            all_json_files.append(os.path.join(directory, json))

    # Create a buffer for failed images
    failed_images = []
    ueyes_img_names = np.unique([name for name in os.listdir(vsgui10k_dir)])

    # Iterate over all images in UEyes
    for ueyes_img_name in tqdm(ueyes_img_names):
        img_width = np.nan
        img_height = np.nan   

        try:
            json_file = [f for f in all_json_files if ueyes_img_name[:-4] in f][0]
            segmentation_compos, segmentation_img = get_segmentation(json_file, return_img=True)
        
            img_height = segmentation_img["shape"][0]
            img_width = segmentation_img["shape"][1]

            for box in segmentation_compos:
                width = np.nan
                height = np.nan
        
                if box["class"] == "Background":# or box["class"] == "Face":
                    continue
                else:
                    width =  box["width"]
                    height =  box["height"]
                    x = (box["column_min"] + width/2) / img_width
                    y = (box["row_min"] + height/2) / img_height

                    segmentation_dict["box_x_center"].append(x)
                    segmentation_dict["box_y_center"].append(y)
                    segmentation_dict["class"].append(box["class"])
                    segmentation_dict["ueyes_img_name"].append(ueyes_img_name)
        
        except:
            failed_images.append(ueyes_img_name)
            continue
        
    segmentation_df = pd.DataFrame(segmentation_dict)
    return segmentation_df, failed_images  

def get_coverage(sample : pd.DataFrame, radius : float, original_height : int, original_width : int, col_y : str, col_x : str) -> float:

    """
    Computes the union of circular coverage areas based on fixations within a specified foveal radius.

    Parameters:
    -----------
    sample : pd.DataFrame
        A pandas DataFrame containing the sample points with columns for x and y coordinates.
    radius : float
        The radius of the circular coverage area around each point.
    original_height : int
        The original height of the coordinate system or area being referenced.
    original_width : int
        The original width of the coordinate system or area being referenced.
    col_y : str
        The column name in the DataFrame that holds the y-coordinate values.
    col_x : str
        The column name in the DataFrame that holds the x-coordinate values.

    Returns:
    --------
    overlap : shapely.geometry.Polygon
        A geometric object representing the union of all circular coverage areas.
    num_points : int
        The number of points used to calculate the coverage areas.
    """
    
    # Initialise buffer
    points = []
    
    # Iterate over all rows in the dataframe and create a circular object of the given radius (fovea)
    for (idx, row) in sample.iterrows():
        x,y = row.loc[col_x] * original_width, row.loc[col_y] * original_height
        point = shapely.geometry.Point(x, y).buffer(radius)
        points.append(point)
    
    # Computes the unary union of all buffered Point objects to obtain the total coverage area.
    overlap = unary_union(points)

    return overlap, len(points)


def get_coverage_mixed(shapes):
    """
    Computes the geometric intersection of all pairwise combinations of the given shapes.

    Parameters:
    -----------
    shapes : list of shapely.geometry objects
        A list of geometric shapes (e.g., polygons, points, lines) for which pairwise intersections will be computed.

    Returns:
    --------
    inter : shapely.geometry.BaseGeometry
        A geometric object representing the union of all pairwise intersections of the input shapes.
    """
    # Computes all pairwise intersections among the input shapes using itertools.combinations.
    # Takes the unary union of these intersections to combine them into a single geometric object.
    inter = unary_union([pair[0].intersection(pair[1]) for pair in itertools.combinations(shapes, 2)])

    return inter

def get_nearest_point(left, top, width, height, x, y, corner=True):
    """
    Finds the nearest point on the perimeter of a rectangle to a given point inside or outside the rectangle.

    Parameters:
    -----------
    left : float
        The x-coordinate of the left side of the rectangle.
    top : float
        The y-coordinate of the top side of the rectangle.
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.
    x : float
        The x-coordinate of the point to compare.
    y : float
        The y-coordinate of the point to compare.
    corner : bool
        Whether to match to closest corner

    Returns:
    --------
    tuple
        A tuple (x, y) representing the nearest point on the perimeter of the rectangle to the given point (x, y).
    """
    # Get right and bottom coordinates
    right = left + width
    bottom = top + height

    # Get closest edge
    xc = np.array([left, right])[np.argmin([(np.abs(x-left)), (np.abs(x-right))])]   
    yc = np.array([top, bottom])[np.argmin([(np.abs(y-top)), (np.abs(y-bottom))])]   

    # Check whether left or right edge is closer to the x-coordinate
    distances = np.array([(abs(yc-top)), (abs(yc-bottom)), (abs(xc-left)), (abs(xc-right))])

    # Check which side is closest to the point
    if corner: # Enable for plots in the paper
        points = np.array([(xc,top), (xc,bottom), (left, yc), (right,yc)])
    else:
        points = np.array([(x,top), (x,bottom), (left, y), (right,y)])
    
    result = points[np.argmin(distances)]

    return result


def plot_heatmaps(basedata: pd.DataFrame, img_type: int, xcol: str, ycol: str, 
                  just_screenshot: bool, normalize: bool, xmax: int, ymax: int, ueyes: bool,
                  first_fixations: bool, category: str, alphabet_index: int, ax: plt.axes, debias: bool) -> int:
    """
    Plot heatmaps based on gaze data.

    Parameters:
    -----------
    basedata : pd.DataFrame
        The base data containing gaze information.
    img_type : int
        The type of image (target description, fixation cross, visual search, validation)
    xcol : str
        Column name for x coordinates in the DataFrame (one of FPOGX, etc.).
    ycol : str
        Column name for y coordinates in the DataFrame (one of FPOGY, etc.).
    just_screenshot : bool
        Whether to filter out gaze data based only on screenshot region.
    normalize : bool
        Whether to normalize the x and y coordinates.
    xmax : int
        Maximum value for x-axis for the heatmap plot.
    ymax : int
        Maximum value for y-axis for the heatmap plot.
    ueyes : bool
        Indicator whether UEyes data is plotted.
    first_fixations : bool
        Whether to include first fixations (FPOGID < 4) (e.g., in visual search windows might bias the fixations to the starting point of the search, the center of the screen).
    category : str
        The category to filter gaze data (mobile, web, desktop); 'all' for no filtering.
    alphabet_index : int
        Index to retrieve a letter from the ASCII uppercase string for labeling the subplot.
    ax : plt.axes
        The matplotlib axes object to plot the heatmap.
    debias : bool
        Whether to use debiased coordinates for plotting (FPOGX_debias and FPOGY_debias).

    Returns:
    --------
    checksum : int
        The number of data points in the filtered dataset.

    """
    
    try:
        # Filter basedata to exclude fixations with FPOGID <= 2
        data = basedata[(basedata.FPOGID > 2)]
        
        # Optionally filter based on img_type
        if not ueyes:
            data = data[(data.img_type == img_type)]
        
        # Optionally filter based on category
        if category != "all":
            data = data[data.category == category]
        
        # Filter based on first fixations criterion
        if first_fixations:
            data = data[(data.FPOGID < 4)]
        else:
            data = data[(data.FPOGID >= 4)]
        
        # Count the number of entries after filtering
        checksum = len(data)
        
        # Use debias coordinates if specified
        if debias:
            _xcol, _ycol = "FPOGX_debias", "FPOGY_debias"
        else:
            _xcol, _ycol = "FPOGX", "FPOGY"
        
        # Filter for screenshot region and normalize if specified
        if just_screenshot:
            
            data = data[(data.FPOGX_scaled > -0.1) 
                        & (data.FPOGX_scaled < 1.1) 
                        & (data.FPOGY_scaled > -0.1)
                        & (data.FPOGY_scaled < 1.1)]
            
            if normalize:
                scaler = MinMaxScaler()
                data["FPOGX_norm"] = scaler.fit_transform(data[["FPOGX_scaled"]])
                data["FPOGY_norm"] = scaler.fit_transform(data[["FPOGY_scaled"]])
        
        # Plot the heatmap using a custom plotting function (plot_heatmap_scipy)
        plot_heatmap_scipy(data, ax, xcol=xcol, ycol=ycol, scatter=False, return_density_max=False, xmax=xmax, ymax=ymax)
        
    except Exception as e:
        print(e)

    # Label the subplot with a letter from ASCII uppercase string
    ax.text(-0.1, 1.1, string.ascii_uppercase[alphabet_index], transform=ax.transAxes, size=12, weight='bold')
    
    # Add dashed lines at midpoints if specified
    if ymax == 1:
        ax.axhline(y=0.5, xmin=0, xmax=1, linestyle="dashed", color="white")
    if xmax == 1:
        ax.axvline(x=0.5, ymin=0, ymax=1, linestyle="dashed", color="white")
    
    # Remove x and y axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Label the x-axis with the number of data points
    ax.set_xlabel(f"$N$={len(data):,}")
    
    return checksum

def plot_three_step(fixations : pd.DataFrame, df_target : pd.DataFrame, FPOGID_min: int, FPOGID_max_lower_bound: int,     FPOGID_max_upper_bound: int, scan_lower: int, scan_higher: int, filename: str, fixation_cross: bool = False) -> None:
    """
    Plots a figure with heatmaps showing different phases of the fixation process.

    Parameters:
    -----------
    FPOGID_min : int
        Minimum fixation identifier to start considering fixations.
    FPOGID_max_lower_bound : int
        Lower bound for the maximum fixation identifier.
    FPOGID_max_upper_bound : int
        Upper bound for the maximum fixation identifier.
    scan_lower : int
        Lower bound for the 'Scan' phase fixations.
    scan_higher : int
        Higher bound for the 'Confirm' phase fixations.
    filename : str
        Name of the file to save the plot.
    fixation_cross : bool, optional
        Whether to include a 'Fixation cross' phase, by default False.

    """

    # Setup the figure and subplots based on fixation_cross parameter
    if fixation_cross:
        fig, axs = plt.subplots(4, 5, figsize=(10, 8))
        phases = ["Example screenshot", "Fixation cross", "Guess", "Scan", "Confirm"]
        col_mapping = {1: f"Fixation cross\n(fixations 1-{FPOGID_min-1})", 
                       2: f"First fixations\n(fixations {FPOGID_min}-{scan_lower-1})", 
                       3: "Intermediate fixations", 
                       4: f"Last fixations\n(last {scan_higher} fixations)"}
    else:
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        phases = ["Example screenshot", "Guess", "Scan", "Confirm"]
        col_mapping = {1: "First fixations", 
                       2: "Intermediate fixations", 
                       3: "Last fixations"}

    # Predefined media IDs for example images
    media_ids = ["83f4bd.png", "314e90.png", "81627a.png", "2304c8.png"]
    sample_images = False
    alphabet_index = 0

    for row, location in enumerate(["upper-left", "upper-right", "lower-right", "lower-left"]):
        np.random.seed(123)
        
        # Filter to include only trials where the target is present and located in the specified location
        sample = fixations[(fixations.absent == False) & (fixations.tgt_location == location)]

        # Select an example image name based on sample_images flag
        if sample_images:
            ueyes_img_name = np.random.choice(np.unique(sample.ueyes_img_name))
        else:
            ueyes_img_name = media_ids[row]

        # Select a target identifier randomly
        tgt_id = np.random.choice(np.unique(sample[sample.img_name == ueyes_img_name].tgt_id))

        # Further filter based on the fixation identifier bounds
        sample = sample[(sample.FPOGID_MAX >= FPOGID_max_lower_bound) & (sample.FPOGID_MAX < FPOGID_max_upper_bound)]

        # Loop through each phase to plot heatmaps
        for col, phase in enumerate(phases):
            ax = axs[row, col]

            if row == 0 and col > 0:
                ax.set_title(f"{col}. {phase}", weight="bold", size=13)
                alphabet_index += 1

            if phase == "Example screenshot":
                # Plot the example screenshot
                img_path = os.path.join("data", "vsgui10k-images", ueyes_img_name)
                img = np.asarray(Image.open(img_path))
                ax.imshow(img)
                ax.set_yticks([])
                ax.set_xticks([])
                _ = draw_target(img_name=ueyes_img_name, tgt_id=tgt_id, img=img, df=df_target, ax=ax, scale=False)

            else:
                # Determine scan phase boundaries and select the appropriate data column
                if scan_lower > 1 and scan_higher > 1:
                    data_col = "FPOGID"
                    _scan_higher = sample.FPOGID_MAX - scan_higher
                else:
                    data_col = "FPOGID_NORM"
                    _scan_higher = scan_higher

                # Filter the sample data based on current phase
                if phase == "Guess":
                    phase_sample = sample[(sample[data_col] >= FPOGID_min) & (sample[data_col] < scan_lower)]
                elif phase == "Scan":
                    phase_sample = sample[(sample[data_col] >= scan_lower) & (sample[data_col] < _scan_higher)]
                elif phase == "Confirm":
                    phase_sample = sample[(sample[data_col] >= _scan_higher)]
                elif phase == "Fixation cross":
                    phase_sample = sample[(sample[data_col] < FPOGID_min)]

                # Plot the heatmap for the current phase
                plot_heatmap_scipy(phase_sample, ax, xcol="FPOGX_scaled", ycol="FPOGY_scaled", scatter=False, return_density_max=False, xmax=1, ymax=1)
                ax.set_yticks([])
                ax.set_xticks([])

                if row == 3:
                    ax.set_xlabel(f"{col_mapping[col]}", size=13)

                # Determine dashed line positions and alignment based on target location
                if "upper" in location:
                    ymin = 0
                    ymax = 0.48
                    alignment = "bottom"
                    y = 0.48

                elif "lower" in location:
                    ymin = 0.52
                    ymax = 1
                    alignment = "top"
                    y = 0.52

                if "left" in location:
                    xmin = 0
                    xmax = 0.48
                    x = 0.48

                elif "right" in location:
                    xmin = 0.52
                    xmax = 1
                    x = 0.52

                # Draw the dashed lines indicating target location
                ax.vlines(x=x, ymin=ymin, ymax=ymax, color="black", linestyle="dashed")
                ax.hlines(y=y, xmin=xmin, xmax=xmax, color="black", linestyle="dashed")

                # Add text annotation for target location
                ax.text(x=xmin + 0.25, y=ymin + 0.25, s="Target\nlocation", horizontalalignment="center", verticalalignment=alignment)

                ax.annotate('', xy=(1, 1), xycoords='axes pixels', arrowprops=dict(facecolor='white', shrink=0.05))

    # Add figure-wide annotations
    fig.text(0.13, 0.885, "Four examples", weight="bold", size=13, horizontalalignment="left")
    if fixation_cross:
        line_coordinates = [0.27, 0.27]
    else:
        line_coordinates = [0.31, 0.31]

    fig.add_artist(lines.Line2D(line_coordinates, [0.1, 0.9], c="black"))
    plt.subplots_adjust(wspace=0.15, hspace=0.01)

    # Ensure the saving directory exists
    save_dir = os.path.join("output", "figs")
    os.makedirs(save_dir, exist_ok=True)

    # Add a text annotation indicating trial length
    fig.text(0.05, 0.05, f"Trial lengths: {FPOGID_max_lower_bound}-{FPOGID_max_upper_bound} fixations", fontstyle="italic", size=10)

    # Save the figure
    fig.savefig(os.path.join(save_dir, f"{filename}"), dpi=300, bbox_inches="tight", pad_inches=0.2)