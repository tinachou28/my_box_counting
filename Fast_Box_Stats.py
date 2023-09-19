#import Py_Box_Count_Stats as PyCount
import numpy as np
import os
try:
    from Numba_Box_Count_Stats import Calc_and_Output_Stats
except ImportError:
    print('Could not import numba functions - numba may not be installed')
    from Box_Count_Stats import Calc_and_Output_Stats


if __name__ == '__main__':
    ################################################
    # set parameters for data
    Lx = 1280 # box size x-dir 
    Ly = 700 # box size y-dir
    Box_Ls = np.array([128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]) # array of box sizes to probe
    #modfile = "./data/noHydro2D_midpoint_run_dt_0.025_nsave_20.suspension_phi_0.34_L_320.config"
    #Nframes = ConvertDataFile(modfile)
    fp = "week2/alice/M5ii-c3/raw/small_no_drift.dat"
    infile = "/Users/tinachou/Desktop/thorneywork 2023/"+fp
    out = "/Users/tinachou/Desktop/thorneywork 2023/week2/alice/M5ii-c3/raw/small out t2"
    outfile = out+"/"
    if not os.path.exists(out):
        os.makedirs(out)
    Nframes = 5000 # number of data frames
    a = 5 #radius of particles
    sep = np.array([2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 4*a, 4*a, 4*a]) #3*a #separation between boxes
    #Calc_MSD_and_Output(infile, outfile, Nframes)
    Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Box_Ls, sep)

    ###########################################################
    # For the cpp module
    #PyCount.Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Box_Ls, sep)
