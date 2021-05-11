import pandas as pd
import numpy as np
import os
from skimage import io


def load_images(root_dir, data_df):
    """A function to load images give a dataframe
    with relative paths
    
    args:
        : root_dir (str): root directory for the chexpert 
        images
        : data_df (pd.DataFrame): pandas dataframe with path names and metadata

    returns:
        : imgs: (list of np.array): an array of images
        : image_meta (pd.DataFrame): data frame of image ids
    """
    imgs = []
    metadata = []
    
    for filepath in data_df['Path']:
        path_split = filepath.split("/")[2:]
        path_join = "/".join(path_split)
        full_path = f"{root_dir}{path_join}"
        
        # structure metdata
        row = {
            'patient_id' : path_split[0],
            'study_number' : path_split[1],
            'image_name' : path_split[2],
        }
        
        metadata.append(row)
        
        # load the image
        imgs.append(io.imread(full_path))
        
    meta = pd.DataFrame(metadata)
    
    return imgs, meta
    
    
        
        
        

        
    
    
    

