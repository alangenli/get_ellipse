# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:42:47 2025

@author: alang
"""

import numpy as np
import pandas as pd
import time
import skimage as ski
from pathlib import Path

import f_ellipse as ELL
import f_graph_ellipse as FIG
    
##############################################################################
#DATA CONSTANTS
##############################################################################
#folder containing image data
in_folder = "DATA_FOLDER"

#extension of the files with the pixel data
data_ext = ".tif"

#SELECT CHANNEL BY INDEX (0, 1, 2)
sel_ch = 2

#light wavelengths for each channel [nm]
ch_wav = [482, 561, 632]
#colours for plotting channels
ch_colour = ['b', 'green', 'r']

##############################################################################
#SCRIPT PARAMETERS
##############################################################################
#size of median filter (integer)
#0 for no median filtering 
#generally not needed if data is clean
#RECOMMENDED DEFAULT: med_filt = 0
med_filt = 1

#set thresh_val = 'auto' to apply automatic thresholding
#otherwise threshold by the specified value
#RECOMMENDED DEFAULT: thresh_val = 'auto'
thresh_val = 0

#CROPPING RANGES IN DATA MASK
#radius of target area, calculated from centre of the image
#measured in pixels
#outside of the ranges, image = 0
#RECOMMENDED DEFAULT: crop_radius = 0
crop_radius = 100

#number of standard deviations used to define outliers in data
#set very high to keep all data
#mainly useful for fitting ellipse
#RECOMMENDED DEFAULT: outlier_std_thresh = 2
outlier_std_thresh = 5

#HOUGH TRANSFORM
#hough transform better for noisy data
#provide a range of radii for fitting a circle
hough_radius_guess = [30, 50]

#WIDTH PARAMETER
#used to define boundaries for data filtering
#RECOMMENDED DEFAULT: width_param = .1
width_param = .1

#SMOOTHING WINDOW for moving average, used for kymographs
#RECOMMENDED DEFAULT: movmean_win = 20
movmean_win = 20

#DEBUGGING MODE
#set to 1 to read only one file even if there are multiple
debug_mode = 0
debug_filename = 'C2_+PIPI2+VAP+Sept-6-2'



#plot bools
plot_raw = 1
plot_crop = 0
plot_med_filt = 0
plot_mask = 1
plot_filt = 1
plot_kymo_ang = 0
plot_kymo_radial = 0
#plot bools for more fancy graph
plot_ellipse_fit = 0
plot_circ_bounds = 1
plot_filt_fancy = 0


#to save model ellipse parameters 
save_kymo = 1

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
    print(f'No files in directory "{in_folder}" with specified extension "{data_ext}".')
    raise SystemExit(0)
else:
    print(f'found {len(file_names)} files in directory "{in_folder}" with desired extension "{data_ext}".')
    #total number of files
    n_im = len(file_names)
    iter_times = n_im*[0]
    ##############################################################################
    if debug_mode:
        #use only one file, for testing
        file_names = [debug_filename]
    ##############################################################################
    
    #FOR ALL FILES
    for n, file_name in enumerate(file_names):
        run_start = time.time()
        """
        READ DATA
        """
        print(f'\nreading file {n+1} of {n_im}: "{file_name}"')
        #use skimage to read file
        im_raw = ski.io.imread(f'{in_folder}/{file_name}{data_ext}')[:,:,sel_ch]
        mask_raw = im_raw>0
        #get image dimensions
        nrow, ncol = np.shape(im_raw)
        #define xy coordinates
        #flip y coordinate so ORIGIN IS AT BOTTOM LEFT
        xcoord, ycoord = np.meshgrid(np.arange(ncol), np.flip(np.arange(nrow)))
        #get all xy coordinate locations where there is a value exceeding the threshold
        xy_data_raw = np.c_[xcoord[mask_raw], ycoord[mask_raw]]
        #create copy
        im_filt = 1*im_raw
        #PLOT RAW IMAGE
        if plot_raw:
            FIG.plot_image(im_raw, title=f"{file_name}\nraw data, ch idx {sel_ch}", colour_map='gray', resolution=100)
        
        """
        CROPPING
        """
        if crop_radius!=0:
            print(f'\nCROPPING IMAGE, target radius = {crop_radius}\n')
            #define circle at centre of image, set all points outside of circle to 0
            im_filt[((np.arange(nrow)[np.newaxis].T - nrow/2)**2 + (np.arange(ncol)[np.newaxis] - ncol/2)**2 > (crop_radius)**2)] = 0
            if plot_crop:
                FIG.plot_image(im_filt>0, title=f"{file_name}, ch idx {sel_ch}\ncropped data mask", colour_map='gray', resolution=100)


        """
        OBTAIN DATA MASK
        """
        if med_filt!=0:
            print("\nAPPLYING MEDIAN FILTER\n")
            #SKIMAGE MEDIAN FILTERING
            im_filt = ski.filters.rank.median(im_filt, ski.morphology.disk(med_filt))
            mask = im_filt>0
            if plot_med_filt:
                FIG.plot_image(mask, title=f"{file_name}, ch idx {sel_ch}\nmedian-filtered data mask", colour_map='gray', resolution=100)
        
        """
        THRESHOLDING
        """
        if thresh_val=='auto':
            #apply threshold value
            mask = im_filt > ski.filters.threshold_otsu(im_filt)
            print(f"\nAUTOMATIC THRESHOLDING, thresh_val = {ski.filters.threshold_otsu(im_raw)}\n")
        else:
            #apply threshold value
            mask = im_filt > thresh_val
            if thresh_val==0:
                print(f"\nthresh_val = {thresh_val} so no threshold applied.\n")
            else:
                print(f"\napplying specified threshold of thresh_val = {thresh_val}\n")
        #update filtered image
        im_filt[~mask] = 0


        """
        REMOVE OUTLIERS FROM DATA MASK
        """
        #get all xy coordinate locations where there is a value exceeding the threshold
        xy_data = np.c_[xcoord[mask], ycoord[mask]]
        #remove outliers
        xy_data, outliers = ELL.remove_outliers_xy(xy_data, outlier_std_thresh)
        #get total length
        nxy = len(xy_data)
        #update filtered image
        im_filt[outliers[:,0], outliers[:,1]] = 0
        #PLOT MASK of DATA with OUTLIERS REMOVED
        if plot_mask:
            #on coordinate grid
            FIG.plot_y(x=xy_data[:,0], 
                       y=xy_data[:,1], 
                       xy_lab=['x', 'y'], 
                       title=f'{file_name}, ch idx {sel_ch}\ndata mask after pre-processing', 
                       colour=ch_colour[sel_ch], marker_style='.', linewidth=0, msize=2)
            FIG.equalise_axes()

        
        """
        HOUGH CIRCLE TRANSFORM
        """
        hough_radii = np.arange(hough_radius_guess[0], hough_radius_guess[1])
        #perform hough circle transform and get parameters
        hough_accums, c_col, c_row, radii =  ski.transform.hough_circle_peaks(ski.transform.hough_circle(im_filt>0, hough_radii), hough_radii, total_num_peaks=1)
        
        """
        DEFINE CIRCLE PARAMETERS FROM HOUGH RADIUS
        """
        #center coordinates
        hough_r = radii[0]
        x0 = c_col[0]
        y0 = nrow-1-c_row[0]
        #upper radius
        r_upper = (1+width_param)*hough_r
        #lower radius
        r_lower = (1-width_param)*hough_r
        
        #RAW FIT, HOUGH FIT
        if plot_ellipse_fit:
            #get hough circle parameters (convert from rowcol to index)
            circ_hough = ELL.circle(r=hough_r, x0 = x0, y0 = y0)
            circ_hough.get_points(num=nxy)
            FIG.plot_xy_N_leg(x=[xy_data[:,0], circ_hough.x], 
                              y=[xy_data[:,1], circ_hough.y], 
                              xy_lab=['x', 'y', f"{file_name}, ch idx {sel_ch}"], 
                              leglabels=['data mask', 'Hough fit'], 
                              altstyle=11, scale=[6, 4.8])
            FIG.equalise_axes()
            

        """
        PLOT UPPER AND LOWER BOUNDARIES
        """
        if plot_circ_bounds:
            #DEFINE LOWER CIRCLE
            circ_l = ELL.circle(r_lower, x0, y0)
            circ_l.get_points(num=nxy)
            #DEFINE UPPER CIRCLE
            circ_h = ELL.circle(r_upper, x0, y0)
            circ_h.get_points(num=nxy)
            #plot
            FIG.plot_xy_N_leg(x=[xy_data[:,0], circ_l.x, circ_h.x], 
                              y=[xy_data[:,1], circ_l.y, circ_h.y], 
                              xy_lab=['x', 'y', f"{file_name}, ch idx {sel_ch}"], 
                              leglabels=['data mask', 'inner circle', 'outer circle'], 
                              altstyle=11, scale=[6, 4.8])
            FIG.equalise_axes()
        
        """
        CENTRE DATA, CALCULATE RADIAL POSITION, ANGLE
        """
        #centre the data from RAW image
        xy_data_centred = xy_data_raw - np.array([x0, y0])
        #calculate radial position (distance from origin)
        data_rpos = np.linalg.norm(xy_data_centred, axis=1)
        #calculate angle
        data_ang = np.arctan2(xy_data_centred[:,1], xy_data_centred[:,0])


        """
        OBTAIN DATA WITHIN CIRCLE BOUNDS
        """
        bool_filt = (data_rpos>=r_lower) & (data_rpos<=r_upper)
        #take data outside of boundaries
        xy_data_rej = xy_data_centred[~bool_filt]
        data_ang_rej = data_ang[~bool_filt]
        data_rpos_rej = data_rpos[~bool_filt]
        #take data within circle boundaries
        xy_data_raw = xy_data_raw[bool_filt]
        xy_data_centred = xy_data_centred[bool_filt]
        data_rpos = data_rpos[bool_filt]
        data_ang = data_ang[bool_filt]
        
        
        #plot filtered data
        if plot_filt:
            #on coordinate grid
            FIG.plot_y(x=xy_data_raw[:,0], 
                       y=xy_data_raw[:,1], 
                       xy_lab=['x', 'y'], 
                       title=f'{file_name}, ch idx {sel_ch}\nmask of retained raw data', 
                       colour=ch_colour[sel_ch], marker_style='.', linewidth=0, msize=2)
            FIG.equalise_axes()
            
        #plot retained and rejected data
        if plot_filt_fancy:
            FIG.plot_xy_N_leg(x=[[xy_data_rej[:,0], xy_data_centred[:,0]], [data_ang_rej, data_ang]], 
                              y=[[xy_data_rej[:,1], xy_data_centred[:,1], ], [data_rpos_rej, data_rpos]], 
                              xy_lab=[['x', 'angle [rad]'], ['y', 'radial position [pixels]'], ['x-y coordinates', 'radial position vs angle', f'{file_name}, ch idx {sel_ch}\nretained and rejected raw data points']], 
                              leglabels=['rejected', 'retained'], 
                              altstyle=28, 
                              scale=[1.3*6, 4.8], ax_legend=1)


        
        """
        KYMOGRAPH
        """
        #CONVERT xy pixel number to index in original image
        idx_row = nrow - xy_data_raw[:,1] - 1
        idx_col = xy_data_raw[:,0]
        #get intensity from the raw image
        im_val = im_raw[idx_row, idx_col]
        
        #combine to single data frame
        kymo_data = pd.DataFrame(np.c_[data_ang, data_rpos, im_val], columns=['ang', 'rpos', 'inten'])
        
        #SAVE DATA
        if save_kymo:
            kymo_data.to_csv(f'{in_folder}/{file_name}_ch_idx{sel_ch}_kymo_data.txt', sep='\t', index=False)
        
        
        #PLOT KYMOGRAPH
        if plot_kymo_ang:
            #sort data
            kymo_data = kymo_data.sort_values('ang')
            #plot intensity vs angle
            FIG.plot_xy_N_leg(x=kymo_data.ang, 
                              y=[kymo_data.inten, ELL.movmean(kymo_data.inten, movmean_win)], 
                              xy_lab=['angle [rad]', 'intensity', f'{file_name}, ch idx {sel_ch}\nangular kymograph'], 
                              leglabels=['data', 'trendline'], 
                              altstyle=14, 
                              scale=[6, 4.8])
        if plot_kymo_radial:
            #sort data
            kymo_data = kymo_data.sort_values('rpos')
            #plot intensity vs radial position
            FIG.plot_xy_N_leg(x=kymo_data.rpos, 
                              y=[kymo_data.inten, ELL.movmean(kymo_data.inten, movmean_win)], 
                              xy_lab=['radial position [pixels]', 'intensity', f'{file_name}, ch idx {sel_ch}\nradial kymograph'], 
                              leglabels=['data', 'trendline'], 
                              altstyle=13, 
                              scale=[6, 4.8])


        #timing
        iter_times[n] = time.time()-run_start
print(f"\ntimes per loop [s]: {iter_times}")
print(f"total elapsed time: {sum(iter_times)}s")
