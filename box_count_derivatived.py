import numpy as np
import os

from Numba_Box_Count_Stats import BoxCountDerivatived


if __name__ == '__main__':
    ################################################
    # set parameters for data
    Lx = 1280 # box size x-dir 
    Ly = 700 # box size y-dir
    fps = 2 #fps of video
    Box_Ls = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25]) # array of box sizes to probe
    #Box_Ls = np.array([128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]) # array of box sizes to probe

    #fp = "week2/eleanor/data_large/retracked subset/data/reindexed/combined.dat"
    fp = "week2/alice/m6iiic4c/raw/small_no_drift.dat"
    infile = "/Users/tinachou/Desktop/thorneywork 2023/"+fp

    Nframes = 7000 # number of data frames
    a = 5 #radius of particles
    sep = np.array([2*a, 2*a, 2*a, 4*a, 4*a, 4*a]) #3*a #separation between boxes

    #sep = np.array([2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 4*a, 4*a, 4*a]) #3*a #separation between boxes
    D = 0.03911 #Diffusion coefficient
    p2m = 0.17 #pixel to um
    linlog = False # want to plot linlog scale?
    BoxCountRaw(infile, Nframes, Lx, Ly, Box_Ls, sep, fps, D, p2m, linlog)
    print("Done.")
    #BoxCountRaw(infile, Nframes, Lx, Ly, np.array([0.5]), np.array([4*a]), fps, D, p2m)
