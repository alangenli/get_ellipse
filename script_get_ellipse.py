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
from bioio import BioImage

import f_ellipse as ELL
import f_graph_ellipse as FIG
    
##############################################################################
#DATA CONSTANTS
##############################################################################
#folder containing image data
in_folder = "DATA_FOLDER"

#extension of the files with the pixel data
data_ext = ".ids"

#SELECT CHANNEL BY INDEX (0, 1, 2)
sel_ch = 1

#light wavelengths for each channel [nm]
ch_wav = [482, 561, 632]

##############################################################################
#SCRIPT PARAMETERS
##############################################################################
#number of standard deviations used to define outliers in data
outlier_std_thresh = 2


#WIDTH PARAMETER
#used to define boundaries for data filtering
width_param = .04


#SMOOTHING WINDOW for moving average, used for kymographs
movmean_win = 20




#plot bools
plot_raw = 1
plot_mask = 0
plot_filt = 0
plot_kymo_ang = 1
plot_kymo_radial = 1
#plot bools for more fancy graphs
plot_ellipse_fit = 0
plot_circ_bounds = 0
plot_filt_fancy = 1


#to save model ellipse parameters 
save_ellipse_param = 0
save_kymo = 1

#DEBUGGING MODE
#set to 1 to read only one file even if there are multiple
debug_mode = 0



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
    iter_times = n_im*[0]
    ##############################################################################
    if debug_mode:
        #use only one file, for testing
        file_names = file_names[-1:]
    ##############################################################################
    
    #FOR ALL FILES
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
            im_raw = ski.io.imread(f'{in_folder}/{file_name}{data_ext}')[:,:,sel_ch]
        #get image dimensions
        nrow, ncol = np.shape(im_raw)
        #PLOT RAW IMAGE
        if plot_raw:
            FIG.plot_image(im_raw, title=f"{file_name}\nraw data, ch idx {sel_ch}", colour_map='gray', resolution=100)
    
        """
        OBTAIN DATA MASK
        """
        #obtain mask using threshold
        mask = im_raw > ski.filters.threshold_otsu(im_raw)
        #define xy coordinates
        #flip y coordinate so ORIGIN IS AT BOTTOM LEFT
        xcoord, ycoord = np.meshgrid(np.arange(ncol), np.flip(np.arange(nrow)))
        #get all xy coordinate locations where there is a value exceeding the threshold
        xy_data = np.c_[xcoord[mask], ycoord[mask]]
        #PLOT MASK of RAW DATA
        if plot_raw:
            #on coordinate grid
            FIG.plot_y(x=xy_data[:,0], 
                       y=xy_data[:,1], 
                       xy_lab=['x', 'y'], 
                       title=f'{file_name}, ch idx {sel_ch}\nraw data mask', 
                       marker_style='.', linewidth=0, msize=2)
            FIG.equalise_axes()
    
        """
        REMOVE OUTLIERS FROM DATA MASK
        """
        #remove outliers
        xy_data = ELL.remove_outliers_xy(xy_data, outlier_std_thresh)
                        
        #PLOT MASK of DATA with OUTLIERS REMOVED
        if plot_mask:
            #on coordinate grid
            FIG.plot_y(x=xy_data[:,0], 
                       y=xy_data[:,1], 
                       xy_lab=['x', 'y'], 
                       title=f'{file_name}, ch idx {sel_ch}\ndata mask, no outliers', 
                       marker_style='.', linewidth=0, msize=2)
            FIG.equalise_axes()

        """
        FIT ELLIPSE TO DATA MASK
        """
        print("fitting ellipse to raw data")
        #get total length
        nxy = len(xy_data)
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
            FIG.plot_xy_N_leg(x=[xy_data[:,0], model_raw.x, model_near.x], 
                              y=[xy_data[:,1], model_raw.y, model_near.y], 
                              xy_lab=['x', 'y', f"{file_name}, ch idx {sel_ch}"], 
                              leglabels=['data mask', 'raw fit', 'nearest fit'], 
                              altstyle=11)
            FIG.equalise_axes()
        
        """
        CENTRE DATA, CALCULATE METRICS
        """
        #centre the data
        xy_data_centred = xy_data - np.array([model_near.x0, model_near.y0])
        #calculate radial position (distance from origin)
        data_rpos = np.linalg.norm(xy_data_centred, axis=1)
        data_ang = np.arctan2(xy_data_centred[:,1], xy_data_centred[:,0])


        """
        DEFINE CIRCLE BOUNDARIES
        """
        #upper radius
        r_upper = (1+width_param)*model_near.major
        #lower radius
        r_lower = (1-width_param)*model_near.minor

        #PLOT UPPER AND LOWER BOUNDARIES
        if plot_circ_bounds:
            #DEFINE LOWER CIRCLE
            circ_l = ELL.circle(r_lower, model_near.x0, model_near.y0)
            circ_l.get_points(num=nxy)
            #DEFINE UPPER CIRCLE
            circ_h = ELL.circle(r_upper, model_near.x0, model_near.y0)
            circ_h.get_points(num=nxy)
            #plot
            FIG.plot_xy_N_leg(x=[xy_data[:,0], circ_l.x, circ_h.x], 
                              y=[xy_data[:,1], circ_l.y, circ_h.y], 
                              xy_lab=['x', 'y', f"{file_name}, ch idx {sel_ch}"], 
                              leglabels=['data mask', 'inner circle', 'outer circle'], 
                              altstyle=11, scale=[6, 4.8])
            FIG.equalise_axes()
        
        """
        FILTER TO OBTAIN DATA WITHIN CIRCLES
        """
        bool_filt = (data_rpos>=r_lower) & (data_rpos<=r_upper)
        #take data outside of boundaries
        xy_data_rej = xy_data_centred[~bool_filt]
        data_ang_rej = data_ang[~bool_filt]
        data_rpos_rej = data_rpos[~bool_filt]
        #take data within circle boundaries
        xy_data = xy_data[bool_filt]
        xy_data_centred = xy_data_centred[bool_filt]
        data_rpos = data_rpos[bool_filt]
        data_ang = data_ang[bool_filt]
        
        
        #plot filtered data
        if plot_filt:
            #on coordinate grid
            FIG.plot_y(x=xy_data[:,0], 
                       y=xy_data[:,1], 
                       xy_lab=['x', 'y'], 
                       title=f'{file_name}, ch idx {sel_ch}\ndata mask, filtered', 
                       marker_style='.', linewidth=0, msize=2)
            FIG.equalise_axes()
            
        #plot retained and rejected data
        if plot_filt_fancy:
            FIG.plot_xy_N_leg(x=[[xy_data_centred[:,0], xy_data_rej[:,0]], [data_ang, data_ang_rej]], 
                              y=[[xy_data_centred[:,1], xy_data_rej[:,1]], [data_rpos, data_rpos_rej]], 
                              xy_lab=[['x', 'angle [rad]'], ['y', 'radial position [pixels]'], ['x-y coordinates', 'radial position vs angle', f'{file_name}, ch idx {sel_ch}\ndata mask, FILTERED data']], 
                              leglabels=['retained', 'rejected'], 
                              altstyle=26, 
                              scale=[1.3*6, 4.8], ax_legend=1)


        
        """
        KYMOGRAPH
        """
        #CONVERT xy pixel number to index in original image
        idx_row = nrow - xy_data[:,1] - 1
        idx_col = xy_data[:,0]
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
