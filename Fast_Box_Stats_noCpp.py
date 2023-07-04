import numpy as np
import scipy.stats as stats

def autocorrFFT(x):
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A


def msd_fft(r):
    if len(r.shape) == 1:
        r = r.reshape(-1, 1)
    N = r.shape[0]
    D=np.square(r)
    D=np.append(D,0) 
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    return S1-2*S2


def Box_Bin_Exp(x, y, Lx, Ly, Lbox, sep):
    Np = len(x)
    SepSize = Lbox + sep
    Nx = int(np.floor(Lx / SepSize))
    Ny = int(np.floor(Ly / SepSize))
    Counts = np.zeros(Nx * Ny)

    for i in range(Np):
        # periodic corrections
        while x[i] > Lx:
            x[i] -= Lx
        while x[i] < 0.0:
            x[i] += Lx
        while y[i] > Ly:
            y[i] -= Ly
        while y[i] < 0.0:
            y[i] += Ly

        # find correct box and increment counts
        II = int(np.floor(x[i] / SepSize))
        JJ = int(np.floor(y[i] / SepSize))
        Xmod = np.fmod(x[i], SepSize)
        Ymod = np.fmod(y[i], SepSize)

        if (II+1.0) * SepSize > Lx:
            continue
        if (JJ+1.0) * SepSize > Ly:
            continue

        if max(np.abs(Xmod-0.5*SepSize), np.abs(Ymod-0.5*SepSize)) < Lbox/2.0:
            Counts[II * Ny + JJ] += 1.0

    return Counts

def outputMatrixToFile(matrix, filename):
    np.savetxt(filename, matrix, delimiter=' ', fmt='%.10f')
    print("Matrix data has been written to", filename)

def computeColumnStats(matrix):
    numRows, numCols = matrix.shape
    sumSqDiff = np.zeros(numCols)
    mean = np.zeros(numCols)
    SEM = np.zeros(numCols)

    for i in range(numRows):
        row_i = matrix[i, :]
        delta = row_i - mean
        mean += (1.0 / (i+1.0)) * delta
        delta2 = row_i - mean
        sumSqDiff += np.multiply(delta, delta2)
    
    SEM = (2.0 / np.sqrt(numRows * (numRows - 1.0))) * np.sqrt(sumSqDiff)
    
    return mean, SEM

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
        print(Ntimes)

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

def processDataFile(filename, Nframes):
    Xs = [[] for _ in range(Nframes)]
    Ys = [[] for _ in range(Nframes)]

    fileinput = open(filename, "r")
    if not fileinput:
        print("Error opening file:", filename)
        return Xs, Ys

    ind, ind_p = 0, 0
    x, y = 0.0, 0.0

    frame = 0
    start = 0
    while True:
        line = fileinput.readline().strip()
        if not line:
            break

        values = line.split()
        x = float(values[0])
        y = float(values[1])
        ind = int(values[2])

        if frame == 0 and ind != 0:
            start = ind
            frame = 1
            ind_p = ind - 1
        if ind_p != ind:
            print(ind)
        
        Xs[ind - start].append(x)
        Ys[ind - start].append(y)
        ind_p = ind

    fileinput.close()

    return Xs, Ys

def processDataFile_and_Count(filename, Nframes, Lx, Ly, Lbox, sep):
    remo = 1
    x, y = processDataFile(filename, Nframes)
    print("Done with data read")

    Counts = [[] for _ in range(len(Lbox))]

    for lbIdx in range(len(Lbox)):
        print("Counting boxes L =", Lbox[lbIdx])
        for nt in range(len(x)):
            Count_nt = Box_Bin_Exp(x[nt], y[nt], Lx, Ly, Lbox[lbIdx], sep[lbIdx])
            Counts[lbIdx].append(Count_nt)

    # convert data to list of matrices
    CountMs = []
    for lbIdx in range(len(Lbox)):
        numCounts = len(Counts[lbIdx]) # number of time steps
        countSize = len(Counts[lbIdx][0]) # number of boxes

        CountM = np.zeros((countSize, numCounts))

        for i in range(numCounts):
            CountM[:, i] = Counts[lbIdx][i]

        CountMs.append(CountM)

    print("Done with counting")
    return CountMs

def computeMeanAndSecondMoment(matrix):
    numRows, numCols = matrix.shape
    n = numRows * numCols

    mean = 0.0
    m2 = 0.0

    for i in range(numRows):
        for j in range(numCols):
            value = matrix[i, j]
            delta = value - mean
            mean += delta / (i * numCols + j + 1.0)
            m2 += delta * (value - mean)

    variance = m2 / n

    alpha = 0.01
    df = 1.0 * matrix.size - 1.0
    chi_lb = stats.chi2.ppf(0.5 * alpha, df)
    chi_ub = stats.chi2.ppf(1.0 - 0.5 * alpha, df)

    variance_sem_lb = (df / chi_lb) * variance
    variance_sem_ub = (df / chi_ub) * variance

    return mean, variance, variance_sem_lb, variance_sem_ub

def Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Lbs, sep):
    CountMs = processDataFile_and_Count(infile, Nframes, Lx, Ly, Lbs, sep)

    N_Stats = np.zeros((len(Lbs), 5))

    for lbIdx in range(len(Lbs)):
        print("Processing Box size:", Lbs[lbIdx])

        N_Stats[lbIdx, 0] = Lbs[lbIdx]
        mean, variance, variance_sem_lb, variance_sem_ub = computeMeanAndSecondMoment(CountMs[lbIdx])
        N_Stats[lbIdx, 1] = mean
        N_Stats[lbIdx, 2] = variance
        N_Stats[lbIdx, 3] = variance_sem_lb
        N_Stats[lbIdx, 4] = variance_sem_ub

        MSDs = np.zeros(CountMs[lbIdx].shape)
        for i in range(CountMs[lbIdx].shape[0]):
            print(100.0 * ((1.0 * i) / (1.0 * MSDs.shape[0])), "percent done with MSD calc")
            MSDrow = CountMs[lbIdx][i, :]
            MSDs[i, :] = msd_fft(MSDrow)  # Assuming msd_fft is defined elsewhere
            # print("MSD:", MSDs[i, :])

        MSDmean = np.mean(MSDs, axis=0)
        MSDsem = np.std(MSDs, axis=0) / np.sqrt(MSDs.shape[0])
        Lstr = format(Lbs[lbIdx], '0.6f')
        outputMatrixToFile(MSDmean, outfile + "_MSDmean_BoxL_" + Lstr + ".txt")
        outputMatrixToFile(MSDsem, outfile + "_MSDerror_BoxL_" + Lstr + ".txt")

    outputMatrixToFile(N_Stats, outfile + "_N_stats.txt")
    
    
if __name__ == '__main__':
    #################################################
    # set parameters for data
    Lx = 288.0 # box size x-dir 
    Ly = 288.0 # box size y-dir
    Box_Ls = np.array([256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]) # array of box sizes to probe
    #modfile = "./data/spec_softetakt_long_run_dtau_0.025_nsave_4.suspension_phi_0.66_L_288.config"
    #ConvertDataFile(modfile)
    infile = "./data/spec_softetakt_long_run_dtau_0.025_nsave_4.suspension_phi_0.66_L_288_modified.txt"
    outfile = "./Count_Data_Cpp/Pure_Py_Test_long_phi_0.66"
    Nframes = 7424 # number of data frames
    a = 1.395 #radius of particles
    sep = np.array([2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 1*a, 1*a]) #3*a #separation between boxes
    Calc_and_Output_Stats(infile, outfile, Nframes, Lx, Ly, Box_Ls, sep)
