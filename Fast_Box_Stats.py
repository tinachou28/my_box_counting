#import Py_Box_Count_Stats as PyCount
import numpy as np
try:
    from Numba_Box_Count_Stats import Calc_and_Output_Stats
except ImportError:
    print('Could not import numba functions - numba may not be installed')
    from Box_Count_Stats import Calc_and_Output_Stats


if __name__ == '__main__':
    ################################################
    # set parameters for data
    Lx = 217.6 # box size x-dir 
    Ly = 174 # box size y-dir
    Box_Ls = np.array([64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]) # array of box sizes to probe
    #modfile = "./data/noHydro2D_midpoint_run_dt_0.025_nsave_20.suspension_phi_0.34_L_320.config"
    #Nframes = ConvertDataFile(modfile)
    infile = "./data/c9allpositionssmall_realunits.dat"
    outfile = "./Count_Data_Cpp/Exp_test" #"./Count_Data_Cpp/New_NH_Py_Test_long_phi_0.34"
    Nframes = 2390 # number of data frames
    a = 1.395 #radius of particles
    sep = np.array([2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 4*a, 4*a, 4*a]) #3*a #separation between boxes
    #Calc_MSD_and_Output(infile, outfile, Nframes)
    Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Box_Ls, sep)
    ###########################################################
    # For the cpp module
    #PyCount.Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Box_Ls, sep)
