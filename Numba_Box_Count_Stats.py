import numpy as np
import scipy.stats as stats
from numba import jit, njit, prange, objmode
from numba.typed import List as nblist
import numba as nb
import sys

###############################
# These two function are from SE
# https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
##############################


def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2*N)  # 2*N because of zero-padding
    PSD = F * np.conjugate(F)
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n  # this is the autocorrelation in convention A

@njit(fastmath=True)
def msd_fft1d(r):
    N = len(r)
    D = np.square(r)
    D = np.append(D, 0)
    with objmode(S2='float64[:]'):
        S2 = autocorrFFT(r)
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in prange(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N-m)
    return S1 - 2 * S2

@njit(parallel=True, fastmath=True)
def msd_matrix(matrix):
    Nrows, Ncols = matrix.shape
    MSDs = np.zeros((Nrows,Ncols))
    for i in prange(Nrows):
        #print(100.0 * ((1.0 * i) / (1.0 * N)), "percent done with MSD calc")
        MSD = msd_fft1d(matrix[i, :])
        MSDs[i,:] = MSD
    return MSDs

@njit(parallel=True, fastmath=True)
def msd_coords(Xs,Ys):
    numRows, numCols = Xs.shape
    MSDs = np.zeros((numRows, numCols))

    for i in prange(numRows):
        xi = Xs[i, :]
        yi = Ys[i, :]
        row_i = msd_fft1d(xi) + msd_fft1d(yi)
        MSDs[i,:] = row_i
    
    return MSDs


def outputMatrixToFile(matrix, filename):
    np.savetxt(filename, matrix, delimiter=' ', fmt='%.10f')
    print("Matrix data has been written to", filename)


def ConvertDataFile(filename):
    fileinput = open(filename, "r")
    if not fileinput:
        print("Error opening file:", filename)
        return

    Ntimes = 0
    x, y, z = [], [], []
    aux1, aux2, aux3, aux4 = 0.0, 0.0, 0.0, 0.0

    # Parse 'filename' to remove the extension (chars after a period) and add the string "_modified.txt" to the result.
    inputFilename = filename
    pos = inputFilename.rfind('.')
    baseFilename = inputFilename[:pos]
    outfile = baseFilename + "_modified.txt"

    fileoutput = open(outfile, "w")
    if not fileoutput:
        print("Error opening output file:", outfile)
        fileinput.close()
        return

    while True:
        line = fileinput.readline().strip()
        if not line:
            break

        parts = int(line)
        Ntimes += 1
        #print(Ntimes)

        x = np.zeros(parts)
        y = np.zeros(parts)

        # Read the data directly into the arrays and write to the output file
        for i in range(parts):
            values = fileinput.readline().split()
            x[i] = float(values[0])
            y[i] = float(values[1])
            fileoutput.write("{:.6f} {:.6f} {}\n".format(x[i], y[i], Ntimes))

    fileinput.close()
    fileoutput.close()
    return Ntimes

# def processDataFile(filename, Nframes):
#     Xs = [[] for _ in range(Nframes)]
#     Ys = [[] for _ in range(Nframes)]

#     fileinput = open(filename, "r")
#     if not fileinput:
#         print("Error opening file:", filename)
#         return Xs, Ys

#     ind, ind_p = 0, 0
#     x, y = 0.0, 0.0

#     frame = 0
#     start = 0
#     while True:
#         line = fileinput.readline().strip()
#         if not line:
#             break

#         values = line.split()
#         x = float(values[0])
#         y = float(values[1])
#         ind = int(values[2])

#         if frame == 0 and ind != 0:
#             start = ind
#             frame = 1
#             ind_p = ind - 1
#         if ind_p != ind:
#             print(ind)
        
#         Xs[ind - start].append(x)
#         Ys[ind - start].append(y)
#         ind_p = ind

#     fileinput.close()
#     return Xs, Ys


def processDataFile(filename, Nframes):
    Xs = [[] for _ in range(Nframes)]
    Ys = [[] for _ in range(Nframes)]

    try:
        with open(filename, "r") as fileinput:
            file_contents = fileinput.readlines()
            ind_p = 0
            for line in file_contents:
                values = line.split()
                try:
                    x = float(values[0])
                    y = float(values[1])
                    ind = round(float(values[2]))
                    Xs[ind-1].append(x)
                    Ys[ind-1].append(y)
                    #if ind_p != ind:
                        #print(ind)
                    ind_p = ind
                except (ValueError, IndexError):
                    print("I can't read index "+str(ind_p)+" of the file")
                    continue
    except IOError:
        print("Error opening file:", filename)
        sys.exit()
    
    return Xs, Ys


@njit(parallel=True, fastmath=True)
def processDataFile_and_Count(x, y, Lx, Ly, Lbox, sep):
    CountMs = nblist()
    for lbIdx in range(len(Lbox)):
        print("Counting boxes L =", Lbox[lbIdx])
        Times = len(x)
        SepSize = Lbox[lbIdx] + sep[lbIdx]
        Nx = int(np.floor(Lx / SepSize))
        Ny = int(np.floor(Ly / SepSize))
        Counts = np.zeros((Nx * Ny, Times), dtype=np.float32)
        for nt in prange(Times):
            xt = x[nt]
            yt = y[nt]
            Np = len(xt)
            for i in range(Np):
                # periodic corrections
                while xt[i] > Lx:
                    xt[i] -= Lx
                while xt[i] < 0.0:
                    xt[i] += Lx
                while yt[i] > Ly:
                    yt[i] -= Ly
                while yt[i] < 0.0:
                    yt[i] += Ly

                # find correct box and increment counts
                II = int(np.floor(xt[i] / SepSize))
                JJ = int(np.floor(yt[i] / SepSize))
                Xmod = np.fmod(xt[i], SepSize)
                Ymod = np.fmod(yt[i], SepSize)

                if (II+1.0) * SepSize > Lx:
                    continue
                if (JJ+1.0) * SepSize > Ly:
                    continue

                if max(np.abs(Xmod-0.5*SepSize), np.abs(Ymod-0.5*SepSize)) < Lbox[lbIdx]/2.0:
                    Counts[II * Ny + JJ,nt] += 1.0
        CountMs.append(Counts)

    print("Done with counting")
    return CountMs


@njit(fastmath=True)
def computeMeanAndSecondMoment(matrix):
    numRows, numCols = matrix.shape
    n = numRows * numCols

    av = 0.0
    m2 = 0.0

    for i in range(numRows):
        for j in range(numCols):
            value = matrix[i, j]
            delta = value - av
            av += delta / (i * numCols + j + 1.0)
            m2 += delta * (value - av)

    variance = m2 / n

    return av, variance

def Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Lbs, sep):
    #CountMs = processDataFile_and_Count(infile, Nframes, Lx, Ly, Lbs, sep)
    Xs,Ys = processDataFile(infile, Nframes)
    print("Done with data read")
    print("Compiling fast counting function (this may take a min. or so)")
    Xnb = nblist(np.array(xi) for xi in Xs)
    Ynb = nblist(np.array(yi) for yi in Ys)
    CountMs = processDataFile_and_Count(Xnb, Ynb, Lx, Ly, Lbs, sep)

    N_Stats = np.zeros((len(Lbs), 5))

    for lbIdx in range(len(Lbs)):
        print("Processing Box size:", Lbs[lbIdx])

        N_Stats[lbIdx, 0] = Lbs[lbIdx]
        #mean, variance, variance_sem_lb, variance_sem_ub = computeMeanAndSecondMoment(CountMs[lbIdx])
        mean_N, variance = computeMeanAndSecondMoment(CountMs[lbIdx])

        ####################
        alpha = 0.01
        df = 1.0 * CountMs[lbIdx].size - 1.0
        chi_lb = stats.chi2.ppf(0.5 * alpha, df)
        chi_ub = stats.chi2.ppf(1.0 - 0.5 * alpha, df)

        variance_sem_lb = (df / chi_lb) * variance
        variance_sem_ub = (df / chi_ub) * variance
        ####################

        N_Stats[lbIdx, 1] = mean_N
        N_Stats[lbIdx, 2] = variance
        N_Stats[lbIdx, 3] = variance_sem_lb
        N_Stats[lbIdx, 4] = variance_sem_ub

        MSDs = msd_matrix(CountMs[lbIdx])

        MSDmean = np.mean(MSDs, axis=0)
        MSDsem = np.std(MSDs, axis=0) / np.sqrt(MSDs.shape[0])
        Lstr = format(Lbs[lbIdx], '0.6f')
        outputMatrixToFile(MSDmean, outfile + "_MSDmean_BoxL_" + Lstr + ".txt")
        outputMatrixToFile(MSDsem, outfile + "_MSDerror_BoxL_" + Lstr + ".txt")

    outputMatrixToFile(N_Stats, outfile + "_N_stats.txt")

 

def Calc_MSD_and_Output(infile, outfile, Nframes):
    #CountMs = processDataFile_and_Count(infile, Nframes, Lx, Ly, Lbs, sep)
    Xs,Ys = processDataFile(infile, Nframes)
    print("Done with data read")
    Xs = np.array(Xs).T
    Ys = np.array(Ys).T
    print("calculating particle MSDs")
    MSDs = msd_coords(Xs,Ys)
    MSDmean = np.mean(MSDs, axis=0)
    MSDsem = np.std(MSDs, axis=0) / np.sqrt(MSDs.shape[0])
    outputMatrixToFile(MSDmean, outfile + "_particles_MSDmean.txt")
    outputMatrixToFile(MSDsem, outfile + "_particles_MSDerror.txt")
