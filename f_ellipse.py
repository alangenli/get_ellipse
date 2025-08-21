# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 11:12:44 2025

@author: alang
"""
import numpy as np
import pandas as pd

##############################################################################
#CLASSES
##############################################################################
class ellipse():
    def __init__(self, a, b, x0, y0, theta):
        """
        CLASS INITIALISATION
        
        ellipse parameters
        """
        self.a = a
        self.b = b
        self.x0 = x0
        self.y0 = y0
        self.theta = theta
        self.major = max([self.a, self.b])
        self.minor = min([self.a, self.b])
    
    def print_val(self):
        print(f"semi-major axis: {self.major}")
        print(f"semi-minor axis: {self.minor}")
        print(f"center coordinate = ({self.x0}, {self.y0})")
        print(f"rotation angle [rad] = {self.theta}")
        print(f"eccentricity = {np.sqrt(1 - self.minor**2/self.major**2)}")
    
    def save_param(self, out_dir):
        param_names = ['a', 'b', 'x0', 'y0', 'theta']
        pd.DataFrame(np.array([self.a, self.b, self.x0, self.y0, self.theta]), index=param_names).to_csv(out_dir, sep='\t', header=False)

        
    def get_points(self, num):
        t = np.linspace(0, 2*np.pi, num)
        self.x = self.a*np.cos(t)*np.cos(self.theta) - self.b*np.sin(t)*np.sin(self.theta) + self.x0
        self.y = self.a*np.cos(t)*np.sin(self.theta) + self.b*np.sin(t)*np.cos(self.theta) + self.y0
    
class circle():
    def __init__(self, r, x0, y0):
        """
        CLASS INITIALISATION
        
        ellipse parameters
        """
        self.r = r
        self.x0 = x0
        self.y0 = y0
    
    def get_points(self, num):
        t = np.linspace(0, 2*np.pi, num)
        self.x = self.r*np.cos(t) + self.x0
        self.y = self.r*np.sin(t) + self.y0

##############################################################################
#ELLIPSE FUNCTIONS
##############################################################################
def get_ellipse_param(A, B, C, D, E):
    """
    get ellipse parameters from least square parameters using the equation
    np.stack([x**2, x*y, y**2, x, y]).T
    NOTE: in this case, F == -1
    """
    discr = B**2 - 4*A*C
    if discr<0:
        a = -1*np.sqrt( 2*(A*E**2 + C*D**2 - B*D*E + -1*discr) * (A + C + np.sqrt((A-C)**2 + B**2)) ) / discr
        b = -1*np.sqrt( 2*(A*E**2 + C*D**2 - B*D*E + -1*discr) * (A + C - np.sqrt((A-C)**2 + B**2)) ) / discr
        x0 = (2*C*D - B*E)/discr
        y0 = (2*A*E - B*D)/discr
        theta = .5*np.arctan2(-B, C-A)
    else:
        print("error! discriminant is greater than or equal to 0, so the parameters do not define an ellipse.")
        raise SystemExit(0)
        
    return a, b, x0, y0, theta


def mat_conic_sec(x, y):
    """
    given x and y coordinates, form the data matrix
    defined by the conic section equation (where the constant F = 1)
    """
    return np.stack([x**2, x*y, y**2, x, y]).T


##############################################################################
#MATH FUNCTIONS
##############################################################################
def remove_outliers_xy(xy_data, std_thresh):
    """
    from array of x-y data, with x and y in separate columns, identify outliers
    outlier defined as points with xy coordinates more than (std_thresh) standard deviations away from the mean
    
    input
    -------
    xy_data, array with arbitrary rows, 2 columns
    std_thresh, integer defining the outlier threshold
    """
    xy_mean = np.mean(xy_data, axis=0)
    xy_std = np.std(xy_data, axis=0)
    xy_data = xy_data[(abs(xy_data[:,0]-xy_mean[0]) < std_thresh*xy_std[0]) & (abs(xy_data[:,1]-xy_mean[1]) < std_thresh*xy_std[1])]
    
    return xy_data
    
'''
def sort_vec(key, X):
    """
    sort all data within X by the key vector
    must have the same dimensions

    """
    idx_sort = np.argsort(key)
    key = key[idx_sort]
    if type(X)==list:
        for n, vec in enumerate(X):
            X[n] = vec[idx_sort]
    else:
        X = X[idx_sort]
    return key, X
'''

def movmean(x, w):
    return np.convolve(x, np.ones(w), mode='same')/w

