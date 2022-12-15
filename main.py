import os
import DataCleaner as dc
from pathlib import Path
import glob
import multiprocessing as mp
import numpy as np

def cleanup_loop(readDir: Path, writeDir: Path, supervised=True, cpuFraction=75):
    """
    Steps through each data file located in readDir and outputs a cleaned-up version in writeDir. Otherwise keeps track of all rejected files.
    supervised sets whether user verification is needed to accept changes or whether it is completed automatically. Unsupervised enables multiprocessing
    using cpuFraction% of all available cores.
    """
    readDir += "\\*.txt"
    writeDir += "\\"
    rejectedFiles = []
    files = glob.glob(readDir)

    #Manual one-by-one checking
    if supervised:
        for file in files:
            rejectedFile = _cleanup_iteration(file, writeDir, supervised=supervised)
            if rejectedFile is not None:
                rejectedFiles.append(rejectedFile)

    #Enabling multiprocessing >:)
    else:
        if cpuFraction > 100 or cpuFraction <= 0:
            raise ValueError("cpuFraction must be between 1-100%")

        cpuCount = mp.cpu_count()
        coresToUse = int(np.ceil((cpuFraction/100)*cpuCount))
        print(f"Using {cpuFraction}% of available cores -> {coresToUse}/{cpuCount}")

        #Creating a tuple of tuples of inputs to pass into each iteration
        writeDirArr = [writeDir]*len(files)
        supervisedArr = [supervised]*len(files)
        args = [*zip(files, writeDirArr, supervisedArr)]

        with mp.Pool(coresToUse) as p:
            p.starmap(_cleanup_iteration, iterable=args)

    print("Cleanup run done!")
    if not len(rejectedFiles) == 0:
        print("Rejected files:")
        for file in rejectedFiles:
            print(file)

def _cleanup_iteration(file: Path, writeDir: Path, supervised=True) -> str|None:
    """
    Internal function which runs an iteration of a cleanup run. Iterated externally by cleanup_loop.
    """
    data = dc.DataCleaner(file)

    fileName = file.split("\\")
    fileName = fileName[len(fileName) - 1]

    ###NOTE: EDIT THIS TO GRAB THE DATAPOINTS YOU NEED FOR A PARTICULAR CLEANUP RUN
    v1 = "Anemometer #1 W Velocity (ms-1)"
    data.remove_nans(v1, data.originalDf)
    data.prune_and(v1, data.std_cutoff(v1, 3), data.gradient_cutoff(v1, 2))

    v2 = "Anemometer #2 W Velocity (ms-1)"
    data.remove_nans(v2, data.originalDf)
    data.prune_and(v2, data.std_cutoff(v2, 3), data.gradient_cutoff(v2, 2))

    t1 = "Anemometer #1 Temperature (degC)"
    data.remove_nans(t1, data.originalDf)
    data.prune_and(t1, data.std_cutoff(t1, 3), data.gradient_cutoff(t1, 2))

    t2 = "Anemometer #2 Temperature (degC)"
    data.remove_nans(t2, data.originalDf)
    data.prune_and(t2, data.std_cutoff(t2, 3), data.gradient_cutoff(t2, 2), iterations=2)
    
    
    #data.plot_comparison(v1, fileName, supervised=supervised, saveLoc=writeDir + "plots")
    #data.plot_comparison(v2, fileName, supervised=supervised, saveLoc=writeDir + "plots")
    #data.plot_comparison(t1, fileName, supervised=supervised, saveLoc=writeDir + "plots")
    #data.plot_comparison(t2, fileName, supervised=supervised, saveLoc=writeDir + "plots")

    #data.plot_ft_dev(v1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
    #data.plot_ft_dev(v2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
    #data.plot_ft_dev(t1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
    #data.plot_ft_dev(t2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
    
    data.plot_ft_loglog(v1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbulent=True, gradient=-5/3)
    data.plot_ft_loglog(v2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbulent=True, gradient=-5/3)
    data.plot_ft_loglog(t1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbulent=True, gradient=-1)
    data.plot_ft_loglog(t2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbulent=True, gradient=-1)

    data.plot_hist(v1, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=100)
    data.plot_hist(v2, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=100)
    data.plot_hist(t1, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=100)
    data.plot_hist(t2, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=100)

    if supervised:
        #Writing cleaned up file or rejecting it
        inputLoop = True
        while inputLoop:
            isAcceptable = input("We happy? [Y/N] ")
            if isAcceptable.lower() == 'y':
                data.df.to_csv(path_or_buf=writeDir + fileName, sep="	")
                print("Yeah. We happy")
                inputLoop = False

            elif isAcceptable.lower() == "n":
                print(f"Rejected {fileName}")
                return fileName
            
            else:
                print("Invalid input. Try again.")

    #If unsupervised, auto-write every time
    else:
        data.df.to_csv(path_or_buf=writeDir + fileName, sep="	")
        
        print(f"Cleaned up {fileName}")

from matplotlib import pyplot as plt
if __name__=='__main__':
    ###NOTE: I/O DIRECTORIES. CHANGE AS REQUIRED
    dir = os.getcwd()
    readDir = dir + "\\Apr2015"
    writeDir = dir + "\\Apr2015_temp_and_w_clean"

    cleanup_loop(readDir, writeDir, supervised=True, cpuFraction=90)
    #data = dc.DataCleaner(writeDir + "\\NRAFBR_02042015_070000.txt")
    #t2 = "Anemometer #2 Temperature (degC)"
    #plt.plot(data.df.GlobalSecs, data.df[t2])
    #plt.show()