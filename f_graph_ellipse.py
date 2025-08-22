"""
@author: alan-gen.li
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Franklin Gothic Medium'})
plt.close('all')


##############################################################################
"""
FUNCTIONS
"""
##############################################################################

def set_xy_labels_N(N, fig, axs, x_lab, y_lab, sel_ax_x_label='sup', sel_ax_y_label=0):
    """
    FUNCTION
    
    to set x and y labels in multi-plot
    varying or shared x or y labels
    
    inputs
    -------
    N, number of plots
    fig, axs, the figure and axes objects
    x_lab, y_lab, the x and y labels (list or string)
    sel_ax_y_label=0
        axes to set shared y label
    """
    #X LABEL
    if len(x_lab)==N:
        #VARIED X LABEL
        for i in range(N): 
            axs[i].set_xlabel(x_lab[i])
    else:
        if sel_ax_x_label=='sup':
            #SHARED X LABEL
            fig.supxlabel(x_lab)
        else:
            #SHARED Y LABEL
            axs[sel_ax_x_label].set_xlabel(x_lab)
    #Y LABEL
    if len(y_lab)==N:
        #VARIED Y LABEL
        for i in range(N): 
            axs[i].set_ylabel(y_lab[i])
    else:
        if sel_ax_y_label=='sup':
            fig.supylabel(y_lab)
        else:
            #SHARED Y LABEL
            axs[sel_ax_y_label].set_ylabel(y_lab)
            
            
def set_titles_N(N, fig, axs, titles):
    """
    FUNCTION
    
    to set titles in multi-plot
    
    unique title per plot
    overall figure title
    
    """
    if len(titles)>=N:
        for i in range(N):
            axs[i].set_title(titles[i])
    if len(titles) % N==1:
        fig.suptitle(titles[-1])
        

def calc_colour_bar(z, colour_map='gist_rainbow', reverse=False):
    """
    FUNCTION
    to calculate colors from data and the colorbar object
    """
    if reverse:
        colour_map +='_r'
    #calculate colours
    if type(z)==list:
        max_colour = max([max(i) for i in z])
        min_colour = min([min(i) for i in z])
        #calculate colour vector
        colours = [getattr(mpl.cm, colour_map)((i-min_colour)/(max_colour-min_colour)) for i in z]
        
    else:
        max_colour = max(z)
        min_colour = min(z)
        #calculate colour vector
        colours = getattr(mpl.cm, colour_map)((z-min_colour)/(max_colour-min_colour))
        
    #map colours to bar
    sm = plt.cm.ScalarMappable(cmap=colour_map, norm=mpl.colors.Normalize(vmin=min_colour, vmax=max_colour))
    
    return colours, sm

def equalise_axes():
    ax=plt.gca()
    ax.set_aspect('equal', adjustable='box')

def add_scatter(x, y, colour='r', msize=5, leg_label=None, leg_colour='k', file_name=0):
    """
    FUNCTION
    to add an additional scatter plot to a previous plot
    """
    ax = plt.gca()
    ax.plot(x, y, label=leg_label, color=colour, linestyle='', marker='o', markersize=msize, fillstyle='none')
    if leg_label!=None:
        ax.legend(framealpha=1, fancybox=False)


##############################################################################
"""
GRAPHS
"""
##############################################################################

def plot_y(y, x = [], xy_lab=[], title=[], colour='b', linewidth=1, marker_style='', msize=5, scale=[.7*6, .7*4.8]):
    """
    FUNCTION
    
    for producing SINGLE plot of y data 
    
    Input:
        y
            if array, then each COLUMN is plotted as separate line
    """
    #plot 
    fig, axs = plt.subplots(figsize=scale)
    axs.spines[['right', 'top']].set_visible(False)
    if len(x)!=0:
        axs.plot(x, y, color=colour, linewidth=linewidth, marker=marker_style, markersize=msize)
    else:
        axs.plot(y, color=colour, linewidth=linewidth, marker=marker_style, markerseize=msize)
        axs.set_xlabel('index')
    #LABELS
    if len(xy_lab)!=0:
        axs.set_xlabel(xy_lab[0])
        axs.set_ylabel(xy_lab[1])
    else:
        axs.set_xlabel('x')
        axs.set_ylabel('y')
    if len(title)!=0:
        axs.set_title(title)
    plt.tight_layout()
    
def plot_image(X, title=0, colour_map = None, resolution=300, file_name=0):
    """
    FUNCTION

    to display an image with varying resolution and colour map
    """
    fig, ax = plt.subplots()
    ax.imshow(X, cmap=colour_map, interpolation='none')
    if title!=0:
        ax.set_title(title)
    ax.set_axis_off()
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=resolution, bbox_inches='tight')  


def plot_xy_N(x, y, xy_lab, altstyle=0, scale=[.7*6, .7*4.8], file_name=0):
    """
    FUNCTION
    
    Plot MULTIPLE x y plots
    
    Input:
        x, list of x vectors
        y, (list) of 2-row matrices
            row 0 is for the line
            row 1 is for the scatter
        xyz_label
            0 is x label
            1 is ylabel
            2 is title (optional)
        leglabels
    """
    if altstyle % 5==0:
        colours = 4*['crimson',  'orangered', 'gold', 'limegreen', 'aqua', 'cornflowerblue', 'blueviolet', 'violet', 'indigo']
    elif altstyle % 5==1:
        colours = ['indigo', 'tab:orange', 'green', 'violet', 'cyan', 'orange', 'indigo', 'pink']
    elif altstyle % 5==2:
        colours = 4*['blue', 'violet', 'red', 'green']
    elif altstyle % 5==3:
        colours = ['limegreen', 'deepskyblue', 'violet', 'crimson']
    elif altstyle % 5==4:
        colours = ['green', 'cornflowerblue', 'limegreen', 'red', 'orangered']
    
    #LINE WITH MARKER
    if altstyle%10 < 5:
        lstyle = '-'
        lwidth = 2
        marker = None
        markersize = 0
    #SCATTER
    elif altstyle%10>=5:
        lstyle = ''
        lwidth = 0
        marker = '.'
        markersize = 5
    N = len(y)
    #SINGLE/ROW PLOTS
    if N==1:
        #SINGLE PLOT
        fig, axs = plt.subplots(sharex=True, figsize=(scale[0], scale[1]))
        axs.plot(x[0], y[0], linestyle=lstyle, linewidth=lwidth, marker = marker, markersize=markersize, color=colours[altstyle])
        axs.set_ylabel(xy_lab[1])
        axs.spines[['top', 'right']].set_visible(False)
        axs.ticklabel_format(style='plain', useOffset=False, axis='y')
        axs.set_xlabel(xy_lab[0])
        #titles
        if len(xy_lab)>=3:
            axs.set_title(xy_lab[-1][0])
    else:
        if altstyle<10:
            #COLUMN PLOT
            fig, axs = plt.subplots(N, 1, sharex=True, figsize=(scale[0], scale[1]))
            for i, ydata in enumerate(y):
                axs[i].plot(x[i], ydata, linestyle=lstyle, linewidth=lwidth, marker = marker, markersize=markersize, color=colours[i])
                axs[i].spines[['top', 'right']].set_visible(False)
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xy_lab[0], xy_lab[1], sel_ax_x_label=-1, sel_ax_y_label='sup')
        else:
            #ROW PLOTS
            #SHARED y axis
            if altstyle>=10 and altstyle<20:
                fig, axs = plt.subplots(1, N, sharex=True, sharey=True, figsize=(scale[0], scale[1]))
            #VARIED y axis
            elif altstyle>=20 and altstyle<30:
                fig, axs = plt.subplots(1, N, sharex=True, figsize=(scale[0], scale[1]))
            #VARIED x and y axis
            elif altstyle>=30:
                fig, axs = plt.subplots(1, N, figsize=(scale[0], scale[1]))
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xy_lab[0], xy_lab[1])
        
        #PLOT LINES
        for i, ydata in enumerate(y):
            axs[i].plot(x[i], ydata, linestyle=lstyle, linewidth=lwidth, marker = marker, markersize=markersize, color=colours[i])
            axs[i].spines[['top', 'right']].set_visible(False)
            axs[i].ticklabel_format(style='plain', useOffset=False, axis='y')
            
        '''
        #PLOT DATA MARKERS
        else:
            for i, ydata in umerate(y):
                axs[i].plot(x[i], ydata, alpha=0.8, marker='.',  markersize=10, color=colours[i])
                axs[i].spines[['top', 'right']].set_visible(False)
                axs[i].ticklabel_format(style='plain', useOffset=False, axis='y')
        '''
        #titles
        if len(xy_lab)>=3:
            set_titles_N(N, fig, axs, xy_lab[2])

    plt.tight_layout(w_pad=1)
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')


def plot_xy_N_leg(x, y, xy_lab, leglabels, altstyle=0, scale=[.7*6, .7*4.8], file_name=0, ax_legend = 0):
    """
    FUNCTION
    
    for N scatter plots
    with legend
    
    Input:
        x, x vector
        y, array of y data matching x
        xy_lab, list of 2 elements
            0 is string of x label
            1 is string for y data
            2 (optional) is plot title
        leglabels, list of strings of labels
        altstyle
            0-19, SINGLE plot
                0-9, shared x axis
                10-19, variable x axis
            >=20, HORIZONTALLY STACKED plots, variable x axis
        file_name, string for saving the plot
        scale, scale of the plots
        save, boolean to save
                    
    """
    if altstyle % 5==0:
        colours = ['crimson',  'orange', 'darkgoldenrod', 'limegreen', 'darkgreen', 'aqua', 'cornflowerblue', 'blueviolet', 'violet', 'indigo']
    elif altstyle % 5==1:
        colours = ['indigo', 'tab:orange', 'green', 'violet', 'cyan']
    elif altstyle % 5==2:
        colours = ['green', 'r', 'deepskyblue', 'indigo','crimson', 'orangered', 'goldenrod']
    elif altstyle % 5==3:
        colours = ['r', 'green', 'indigo', 'deepskyblue', 'crimson', 'orangered', 'goldenrod']
    elif altstyle % 5==4:
        colours = ['lightskyblue', 'b', 'k']
    if altstyle%20>=5 and altstyle%20<10:
        lstyles = 10*[None]
        lwidths = 10*[0]
        markers = 10*['.']
        markersizes = 10*[5]
    elif altstyle%20>=10 and altstyle%20<15:
        lstyles = ['', '--', '-']
        lwidths = [0, 2, 2]
        markers = ['.', '', '']
        markersizes = [2, 0, 0]
    else:
        lstyles = 10*['-']
        lwidths = 10*['1']
        markers = 10*['']
        markersizes = 10*[0]
    if altstyle<20:
        #SINGLE PLOT
        fig, axs = plt.subplots(figsize=(scale[0], scale[1]))
        axs.spines[['top', 'right']].set_visible(False)
        #SHARED X VECTOR
        if len(x)!=len(y):
            for j, ydata in enumerate(y):
                axs.plot(x, ydata, label=leglabels[j], linestyle=lstyles[j], linewidth=lwidths[j], marker = markers[j], markersize=markersizes[j], color=colours[j], alpha=.75)
        #VARYING X VECTOR
        else:
            for j, ydata in enumerate(y):
                axs.plot(x[j], ydata, label=leglabels[j], linestyle=lstyles[j], linewidth=lwidths[j], marker = markers[j], markersize=markersizes[j], color=colours[j], alpha=.75)
    
        axs.set_xlabel(xy_lab[0])
        axs.set_ylabel(xy_lab[1])
        #LEGEND
        if len(leglabels)>len(y):
            axs.legend(frameon=False, title=leglabels[-1])
        else:
            axs.legend(frameon=False)
        #TITLE
        if len(xy_lab)==3:
            axs.set_title(xy_lab[2])
    elif altstyle>=20:
        #HORIZONTAL STACK, SINGLE LEGEND
        N = len(y)
        fig, axs = plt.subplots(1, N, figsize=(scale[0], scale[1]))
        for i in range(N):
            #VARIED X VECTOR
            if len(x[i])==len(y[i]):
                for j, ydata in enumerate(y[i]):
                    axs[i].plot(x[i][j], ydata, label=leglabels[j], linestyle=lstyles[j], linewidth=lwidths[j], marker = markers[j], markersize=markersizes[j], color=colours[j])
                    axs[i].spines[['top', 'right']].set_visible(False)
            #SHARED x vector
            else:
                for j, ydata in enumerate(y[i]):
                    axs[i].plot(x[i], ydata, label=leglabels[j], linestyle=lstyles[j], linewidth=lwidths[j], marker = markers[j], markersize=markersizes[j], color=colours[j])
                    axs[i].spines[['top', 'right']].set_visible(False)
        
        #SET X Y LABELS
        set_xy_labels_N(N, fig, axs, xy_lab[0], xy_lab[1])

        #LEGEND
        if len(leglabels)>len(y[0]):
            #LEGEND WITH TITLE
            axs[ax_legend].legend(frameon=False, title=leglabels[-1])
        else:
            #LEGEND NO TITLE
            axs[ax_legend].legend(frameon=False)
        #TITLES
        if len(xy_lab)>=3:
            set_titles_N(N, fig, axs, xy_lab[2])

    plt.tight_layout(w_pad=1)
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')


def plot_xy_N_colour(x, y, z, xyz_label, altstyle=0, scale=[.7*6, .7*4.8], file_name=0, reverse_colour=False):
    """
    FUNCTION
    
    for N horizontally or vertically stacked plots
    SHARED COLOUR BAR
    
    Input:
        x, x vector
        y, N element list or matrix containing ROW vectors of y data
        z, single vector or LIST of vector(s) by which to colour the data
        xyz_label
            0 is the xlabels for all plots
            1 is the ylabel for all plots
            2 is colourbar label
            3 is list of titles
        altstyle
            if <10 then single plot
            mod 5==0 then rainbow
            mod 5==1 then jet
        scale, scale of the plots
        file_name, string for saving the plot
    """
    N = len(y)

    #define colour vector
    if altstyle % 5==0:
        colours, sm = calc_colour_bar(z, 'gist_rainbow', reverse_colour)
    elif altstyle % 5==1:
        colours, sm = calc_colour_bar(z, 'jet', reverse_colour)
    elif altstyle % 5==2:
        colours, sm = calc_colour_bar(z, 'cool', reverse_colour)
    elif altstyle % 5==3:
        colours, sm = calc_colour_bar(z, 'viridis', reverse_colour)
    elif altstyle % 5==4:
        colours, sm = calc_colour_bar(z, 'YlOrRd', reverse_colour)
    if altstyle % 10 < 5:
        l_style='-'
        m_style = ''
        m_size = 1
    elif altstyle % 10 >= 5:
        l_style='-'
        m_style = 'o'
        m_size = 5
    #SINGLE PLOT
    if altstyle<10:
        fig, axs = plt.subplots(figsize = scale)
        axs.spines[['top', 'right']].set_visible(False)
        #SHARED X VECTOR
        if len(x)!=len(y):
            for j, ydata in enumerate(y):
                axs.plot(x, ydata, alpha=0.75, linestyle=l_style, marker = m_style, markersize=m_size, color=colours[j])
        #VARYING X VECTOR
        else:
            for j, ydata in enumerate(y):
                axs.plot(x[j], ydata, alpha=0.75, linestyle=l_style, marker = m_style, markersize=m_size, color=colours[j])
        #LABELS
        axs.set_xlabel(xyz_label[0])
        axs.set_ylabel(xyz_label[1])
        #TITLE
        if len(xyz_label)>=4:
            axs.set_title(xyz_label[3])
    #SINGLE ROW OR COLUMN
    elif altstyle>=10 and altstyle<50:
        #HORIZONTAL STACK
        if altstyle>=10 and altstyle<40:
            #SHARED X AND Y TICKS, LABELS
            if altstyle<20:
                fig, axs = plt.subplots(1, N, sharex=True, sharey=True, figsize = scale)
            #VARIED AXIS
            elif altstyle>=20 and altstyle<30:
                fig, axs = plt.subplots(1, N, sharey=True, figsize = scale)
            elif altstyle>=30 and altstyle<40:
                fig, axs = plt.subplots(1, N, figsize = scale)
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xyz_label[0], xyz_label[1])
        #VERTICAL STACK
        elif altstyle>=40:
            fig, axs = plt.subplots(N, 1, sharex=True, figsize = scale)
            axs[-1].set_xlabel(xyz_label[0])
            for i in range(N):
                axs[i].set_ylabel(xyz_label[1][i])
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xyz_label[0], xyz_label[1], sel_ax_x_label=-1, sel_ax_y_label='sup')
            
        #PLOT
        for i in range(N):
            axs[i].spines[['top', 'right']].set_visible(False)
            #VARYING X VECTOR
            if len(x[i])==len(y[i]) and type(x[i][0]) is not float:
                for j, ydata in enumerate(y[i]):
                    axs[i].plot(x[i][j], ydata, alpha=0.75, linestyle=l_style, marker = m_style, markersize=m_size, color=colours[i][j])
            #SHARED X VECTOR
            else:
                for j, ydata in enumerate(y[i]):
                    axs[i].plot(x[i], ydata, alpha=0.75, linestyle=l_style, marker = m_style, markersize=m_size, color=colours[i][j])
        
        #TITLE
        if len(xyz_label)>=4:
            set_titles_N(N, fig, axs, xyz_label[3])

    plt.tight_layout(w_pad=2)
    #COLOUR BAR
    cbar = plt.colorbar(sm, ax=axs, pad = 0.02)
    #cbar = plt.colorbar(sm, ax=axs, aspect=25, pad=0.2)
    cbar.set_label(xyz_label[2])
    #save figure
    if file_name!=0:
        fig.savefig(file_name+'.png', dpi=300, bbox_inches='tight')
        
'''

def set_xy_lims(x_lims=0, y_lims=0, file_name=0):
    ax = plt.gca()
    if x_lims!=0:
        ax.set_xlim(x_lims)
    if y_lims!=0:
        ax.set_ylim(y_lims)
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')  

def add_right_yaxis(x, y, ylabel, colour='r', file_name=0):
    """
    FUNCTION
    
    to add a secondary right-side yaxis to a SINGLE plot
    """
    ax1 = plt.gca()
    ax1colour = ax1.get_lines()[-1].get_c()
    ax1.tick_params(axis='y',  labelcolor=ax1colour)
    ax1.spines['left'].set(color=ax1colour)
    
    ax2 = ax1.twinx()
    ax2.plot(x, y, alpha=.75, color=colour)
    ax2.set_ylabel(ylabel)
    ax2.tick_params(axis='y', labelcolor=colour)
    ax2.spines['right'].set(color=colour)
    ax2.spines[['left','top']].set_visible(False)
    plt.tight_layout()
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')    


def invert_xaxis(sel_ax=0, file_name=0):
    """
    FUNCTION
    
    to invert x axis
    
    inputs
    -------
    sel_ax, index to select axes to invert
        DEAFULT 0
        = 'all' to invert all axes
    """
    if sel_ax=='all':
        for i in range(len(plt.gcf().get_axes())):
            plt.gcf().get_axes()[i].xaxis.set_inverted(True)
    else:
        plt.gcf().get_axes()[sel_ax].xaxis.set_inverted(True)
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')    









def ex_labels(lab_codes, titles, key):
    """
    FUNCTION
    to find string(s) in list containing the key
    """
    lab = [i for i in lab_codes if key in i]
    titles = [titles[lab_codes.index(i)] for i in lab]
    num = len(lab)
    return lab, titles, num

'''