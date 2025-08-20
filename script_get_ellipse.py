# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:42:47 2025

@author: alang
"""

import numpy as np
import time
import skimage as ski
from pathlib import Path
from bioio import BioImage

import f_ellipse as ELL
import f_graph_ellipse as FIG
    
##############################################################################
#CONSTANTS
##############################################################################
#folder containing image data
in_folder = "DATA_FOLDER"

#extension of the files with the pixel data
data_ext = ".ids"

#SELECT CHANNEL BY INDEX (0, 1, 2)
sel_ch = 1

#WIDTH PARAMETER
#used to define boundaries for data filtering
width_param = .04


#plot bools
plot_raw = 0
plot_mask = 0
plot_ellipse_fit = 0
plot_circ_bounds = 0
plot_filt = 1
plot_kymo = 1

#to save model ellipse parameters 
save_ellipse_param = 0

#DEBUGGING MODE
#set to 1 to read only one file even if there are multiple
debug_mode = 0



##############################################################################
#SCRIPT
##############################################################################
file_names = list(Path(f'./{in_folder}').glob('*'))
if len(file_names)==0:
    print('error! directory is incorrect or empty.')
else:    
    #extract file names in data folder (specified data extension ONLY)
    file_names = [x.stem for x in file_names if x.suffix==data_ext]
    
if len(file_names)==0:
    print(f'No files in directory "{in_folder}" with specified extension {data_ext}.')
else:
    print(f'found {len(file_names)} files in directory "{in_folder}" with desired extension {data_ext}.')
    #total number of files
    n_im = len(file_names)
    iter_times = n_im*[0]
    ##############################################################################
    if debug_mode:
        #use only one file, for testing
        file_names = file_names[-1:]
    ##############################################################################
    for n, file_name in enumerate(file_names):
        run_start = time.time()
        """
        READ DATA
        """
        print(f"reading file {n+1} of {n_im}")
        if data_ext in ['.ics', '.ids']:
            #use bioio BioImage to read ICS/IDS file
            im_raw = np.squeeze(BioImage(f'{in_folder}/{file_name}{data_ext}').data)[sel_ch,:,:]
            print("\n[***END OF BIOIMAGE OUTPUT***]\n")
        else:
            #use skimage to read file
            im_raw = ski.io.imread(f'{in_folder}/{file_name}{data_ext}')
        #normalize values from 0 to 1
        #im_raw = ELL.normalize(im_raw)
        #get image dimensions
        nrow, ncol = np.shape(im_raw)
        #PLOT RAW IMAGE
        if plot_raw:
            FIG.plot_image(im_raw, title="raw data", colour_map='gray', resolution=100)
    
        """
        OBTAIN DATA MASK
        """
        #obtain mask using threshold
        mask = im_raw > ski.filters.threshold_otsu(im_raw)
        #PLOT MASK
        if plot_mask:
            FIG.plot_image(mask, title="data mask", colour_map='gray', resolution=100)
    
        #every coordinate at which to evaluate the interpolation
        #flip y coordinate so ORIGIN IS AT BOTTOM LEFT
        xcoord, ycoord = np.meshgrid(np.arange(ncol), np.flip(np.arange(nrow)))
        #get all xy coordinate locations where there is a value exceeding the threshold
        xy_data = np.c_[xcoord[mask], ycoord[mask]]
        nxy = len(xy_data)
        

        """
        FIT ELLIPSE TO RAW DATA
        """
        #calculate least squares parameters
        param_raw = np.linalg.lstsq(ELL.mat_conic_sec(*xy_data.T), np.ones((nxy,1)))
        #create ellipse model
        model_raw = ELL.ellipse(*np.squeeze(ELL.get_ellipse_param(*param_raw[0])))
        #get xy coordinates from ellipse model
        model_raw.get_points(num=nxy)

        """
        FIND NEAREST DATA POINTS TO ELLIPSE MODEL
        """
        #identify the point within the original data CLOSEST to the ellipse model
        nearest_data = xy_data[ [np.argmin( np.linalg.norm(xy_data - np.array([model_raw.x[i], model_raw.y[i]]), axis=1) ) for i in range(nxy)] ]
        
        """
        FIT ELLIPSE TO NEAREST DATA POINTS
        """
        #calculate least squares parameters
        param_near = np.linalg.lstsq(ELL.mat_conic_sec(*nearest_data.T), np.ones((nxy,1)))
        #create ellipse model
        model_near = ELL.ellipse(*np.squeeze(ELL.get_ellipse_param(*param_near[0])))
        model_near.get_points(num=nxy)
        print(f"{file_name}")
        print("\nNEAREST FIT ELLIPSE PARAMETERS\n[measured in pixels]")
        model_near.print_val()
        if save_ellipse_param:
            #save parameters to txt file
            model_near.save_param(f'{in_folder}/{file_name}_ellipse_fit.txt')

        #RAW FIT, NEAREST DATA POINTS, NEAREST FIT
        if plot_ellipse_fit:
            FIG.plot_xy_N_leg(x=[xy_data[:,0], model_raw.x, model_near.x], y=[xy_data[:,1], model_raw.y, model_near.y], xy_lab=['x', 'y', f"{file_name}"], leglabels=['data mask', 'raw fit', 'nearest fit'], altstyle=11)
            FIG.equalise_axes()
        
        """
        CENTRE DATA, CALCULATE METRICS
        """
        #centre the data
        xy_data_centred = xy_data - np.array([model_near.x0, model_near.y0])
        #calculate magnitude (distance from origin)
        data_mag = np.linalg.norm(xy_data_centred, axis=1)
        data_ang = np.atan2(xy_data_centred[:,1], xy_data_centred[:,0])
        if plot_mask:
            FIG.plot_xy_N(x=[xy_data_centred[:,0], data_ang], y=[xy_data_centred[:,1], data_mag], xy_lab=[['x', 'angle [rad]'], ['y', 'magnitude'], ['x-y coordinates', 'distance from origin vs angle', f'{file_name}\ndata mask centred at modelled ellipse origin']], scale=[.8*1.5*6, .8*4.8], altstyle=36)


        """
        DEFINE CIRCLE BOUNDARIES
        """
        #upper radius
        r_upper = (1+width_param)*model_near.major
        #lower radius
        r_lower = (1-width_param)*model_near.minor

        #RAW FIT, NEAREST DATA POINTS, NEAREST FIT
        if plot_circ_bounds:
            #DEFINE LOWER CIRCLE
            circ_l = ELL.circle(r_lower, model_near.x0, model_near.y0)
            circ_l.get_points(num=nxy)
            #DEFINE UPPER CIRCLE
            circ_h = ELL.circle(r_upper, model_near.x0, model_near.y0)
            circ_h.get_points(num=nxy)
            #plot
            FIG.plot_xy_N_leg(x=[xy_data[:,0], circ_l.x, circ_h.x], y=[xy_data[:,1], circ_l.y, circ_h.y], xy_lab=['x', 'y', f"{file_name}"], leglabels=['data mask', 'inner circle', 'outer circle'], altstyle=11)
            FIG.equalise_axes()
        
        """
        FILTER TO OBTAIN DATA WITHIN CIRCLES
        """
        bool_filt = (data_mag>=r_lower) & (data_mag<=r_upper)
        #take data within circle boundaries
        xy_data = xy_data[bool_filt]
        xy_data_centred = xy_data_centred[bool_filt]
        data_mag = data_mag[bool_filt]
        data_ang = data_ang[bool_filt]
        
        #plot filtered data
        if plot_filt:
            FIG.plot_xy_N(x=[xy_data_centred[:,0], data_ang], y=[xy_data_centred[:,1], data_mag], xy_lab=[['x', 'angle [rad]'], ['y', 'magnitude'], ['x-y coordinates', 'distance from origin vs angle', f'{file_name}\ndata mask, FILTERED data']], scale=[.8*1.5*6, .8*4.8], altstyle=36)

        
        """
        KYMOGRAPH
        """
        #CONVERT xy pixel number to index in original image
        idx_row = nrow - xy_data[:,1]
        idx_col = xy_data[:,0]
        im_filt = im_raw[idx_row, idx_col]
        #PLOT KYMOGRAPH
        if plot_kymo:
            FIG.plot_xy_N_colour([data_ang, xy_data[:,0], data_mag], 3*[im_filt], 3*[data_mag], [['angle [rad]', 'x', 'magnitude'], 'image intensity', 'distance from origin', [f'{file_name}\nimage intensity vs angle, x-coordinate, and distance from origin']], altstyle=26, scale=[2*6.4, 4.8])
            
        iter_times[n] = time.time()-run_start
        
print(f"\ntimes per loop [s]: {iter_times}")
print(f"total elapsed time: {sum(iter_times)}s")
