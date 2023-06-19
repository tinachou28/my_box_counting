import Py_Box_Count_Stats as PyCount
import numpy as np


if __name__ == '__main__':
    #########################################################
    # set parameters for data
    Lx = 320.0 # box size x-dir TODO: The C++ code is only setup to take Lp but it works with both
    Ly = 320.0 # box size y-dir
    Box_Ls = np.array([64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]) # array of box sizes to probe
    infile = "./data/DfromPhi0.2_noHydro2D_nosteric_run_dt_0.003125_nsave_160.suspension_phi_0.34_L_320_modified.txt"
    outfile = "./Count_Data_Cpp/No_Hydro_Py_Test_phi_0.34"
    Nframes = 11830 # number of data frames
    #infile = "./data/spec_softetakt_long_run_dtau_0.025_nsave_2.suspension_phi_0.34_L_320_modified.txt"
    #outfile = "./Count_Data_Cpp/Py_Test_phi_0.34" #
    #Nframes = 26478 # number of data frames
    a = 1.395 #radius of particles
    sep = np.array([2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 4*a, 4*a, 4*a]) #3*a #separation between boxes
    PyCount.Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Box_Ls, sep)
    
    # set parameters for data
    #Lx = 128.0 # box size x-dir 
    #Ly = 128.0 # box size y-dir
    #Box_Ls = np.array([64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]) # array of box sizes to probe
    ##modfile = "./data/spec_softetakt_run_dtau_0.025_nsave_4.suspension_phi_0.66_L_128.config"
    ##PyCount.ConvertDataFile(modfile)
    #infile = "./data/spec_softetakt_run_dtau_0.025_nsave_4.suspension_phi_0.66_L_128_modified.txt"
    #outfile = "./Count_Data_Cpp/Py_Test_phi_0.66"
    #Nframes = 2500 # number of data frames
    ##infile = "./data/spec_softetakt_long_run_dtau_0.025_nsave_2.suspension_phi_0.34_L_320_modified.txt"
    ##outfile = "./Count_Data_Cpp/Py_Test_phi_0.34" #
    ##Nframes = 26478 # number of data frames
    #a = 1.395 #radius of particles
    #sep = np.array([2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 1*a, 1*a, 0.5*a, 0.25*a]) #3*a #separation between boxes
    #PyCount.Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Box_Ls, sep)
    
    ###############################################
    #Lx = 800.0 # box size x-dir 
    #Ly = 800.0 # box size y-dir
    #Box_Ls = np.array([64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]) # array of box sizes to probe
    ##modfile = "./data/spec_softetakt_run_dtau_0.025_nsave_2.suspension_phi_0.02_L_800.config"
    ##PyCount.ConvertDataFile(modfile)
    #infile = "./data/spec_softetakt_run_dtau_0.025_nsave_2.suspension_phi_0.02_L_800_modified.txt"
    #outfile = "./Count_Data_Cpp/Py_Test_phi_0.02"
    #Nframes = 2900 # number of data frames
    ##infile = "./data/spec_softetakt_long_run_dtau_0.025_nsave_2.suspension_phi_0.34_L_320_modified.txt"
    ##outfile = "./Count_Data_Cpp/Py_Test_phi_0.34" #
    ##Nframes = 26478 # number of data frames
    #a = 1.395 #radius of particles
    #sep = np.array([2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 1*a, 1*a, 0.5*a, 0.25*a]) #3*a #separation between boxes
    #PyCount.Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Box_Ls, sep)
    
    
