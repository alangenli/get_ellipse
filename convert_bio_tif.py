# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 12:28:10 2025

@author: alang
"""
import numpy as np
from bioio import BioImage
import skimage as ski
from pathlib import Path

##############################################################################
#DATA CONSTANTS
##############################################################################

#folder containing image data
in_folder = "DATA_FOLDER"

#extension of the files with the pixel data
data_ext = ".ids"



##############################################################################
#SCRIPT
##############################################################################
file_names = list(Path(f'./{in_folder}').glob('*'))
if len(file_names)==0:
    print('error! directory is incorrect or empty.')
    raise SystemExit(0)
else:    
    #extract file names in data folder (specified data extension ONLY)
    file_names = [x.stem for x in file_names if x.suffix==data_ext]
    
if len(file_names)==0:
    print(f'No files in directory "{in_folder}" with specified extension {data_ext}.')
    raise SystemExit(0)
else:
    print(f'found {len(file_names)} files in directory "{in_folder}" with desired extension {data_ext}.')
    #total number of files
    n_im = len(file_names)
    #FOR ALL FILES
    for n, file_name in enumerate(file_names):
        """
        READ DATA
        """
        print(f"reading file {n+1} of {n_im}")
        if data_ext in ['.ics', '.ids']:
            #use bioio BioImage to read ICS/IDS file
            im_raw = np.squeeze(BioImage(f'{in_folder}/{file_name}{data_ext}').data)
            print("\n[***END OF BIOIMAGE OUTPUT***]\n")
            
            ski.io.imsave(f'{in_folder}/{file_name}.tif', im_raw)