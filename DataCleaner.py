from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from fractions import Fraction

class DataCleaner:
    def __init__(self, dir: Path) -> None:
        #Reading in file
        file = open(dir, "r")
        lines = file.readlines()

        if lines[0] == "% NRA Flarebridge Research Data provided by Woodside Energy Ltd, collected by RPS OceanMonitor System v1.3\n":
            #Grabbing headers
            columnNames = lines[3].split("	")
            units = lines[4].split("	")

            tempColumns = []
            #We give special treatment to the first and last elements to avoid grabbing %Units and the like (these two entries don't have units anyway)
            #We also chop off parts of the string in these two cases to avoid % and \n
            tempColumns.append(columnNames[0][2:])
            for i in range(1, len(columnNames) - 1):
                #Avoiding adding spaces if the unit is dimensionless
                if units[i] == "":
                    tempColumns.append(columnNames[i])
                else:
                    tempColumns.append(columnNames[i] + " (" + units[i] + ")")
            tempColumns.append(columnNames[len(columnNames) - 1][:len(columnNames[len(columnNames) - 1]) - 1])
            
            self.df = pd.read_csv(dir, sep = "	", skiprows=7)
            self.df.columns = tempColumns

            #Adding in a global second entry to avoid having to deal with modular seconds
            globSeconds = self.df.Second.add(60*self.df.Minute)
            self.df.insert(3, "GlobalSecs", globSeconds)
        
        #If the file has previously been cleaned, we can just directly read from it without any tidying required
        else:
            self.df = pd.read_csv(dir, sep = "	")

        #Handy values
        self.mean = self.df.mean()
        self.std = self.df.std()

        #Keeping an unedited copy for reference
        self.originalDf = self.df.copy(deep=True)

        #Chopping off erroneous endpoints where time resets
        self.df = self.df.loc[self.df.index < len(self.df) - 1]
        self.originalDf = self.originalDf.loc[self.originalDf.index < len(self.originalDf) - 1]

        file.close()

    def plot_comparison(self, entry: str, fileName: str, supervised=False, saveLoc=None, plotTitle="_COMPARISON_", plotType="-o") -> None:
        """
        Presents a plot of the values removed from entry during cleanup in the .txt fileName. plotType specifies the plotting marker.
        """
        x = self.df.GlobalSecs

        yChanged = self.df[entry]
        yOriginal = self.originalDf[entry]

        title = fileName[:len(fileName) - 5] + plotTitle + entry

        plt.plot(x, yOriginal, plotType, color='r')
        plt.plot(x, yChanged, plotType, color='b')
        plt.xlabel('GlobalSecs')
        plt.ylabel(entry)
        plt.title(title)

        if supervised:
            plt.show()
        else:
            plt.savefig(saveLoc + "\\" + title + ".png")
            plt.close()

    def plot_ft_dev(self, entry: str, fileName: str, supervised=False, saveLoc=None, plotTitle="_FT_DEVIATION_", plotType="-o", sampleSpacing=1) -> None:
        """
        Presents a plot of the % deviation in FFT as a result of the cleanup. sampleSpacing is the spacing between points in
        the frequency domain. Refer to plot_comparison for parameter details.
        """
        N = len(self.df[entry])
        ft_x = fftfreq(N, sampleSpacing)[:N//2] #The function is symmetric so we are only interested in +ive frequency values

        ft_yChanged = fft(self.df[entry].values)
        ft_yChanged = 2/N * np.abs(ft_yChanged[:N//2])**2 #Multiplying by 2 to deal with the fact that we chopped out -ive freqs
        ft_yChanged[np.abs(ft_yChanged) < 1e-2] = 1e-2 #Capping miniscule values off to prevent deviation from blowing up

        ft_yOriginal = fft(self.originalDf[entry].values)
        ft_yOriginal = 2/N * np.abs(ft_yOriginal[:N//2])**2
        ft_yOriginal[np.abs(ft_yOriginal) < 1e-2] = 1e-2
        
        dev = 100*np.abs((ft_yChanged - ft_yOriginal)/ft_yOriginal)

        title = fileName[:len(fileName) - 5] + plotTitle + entry

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Frequency Domain')

        ax1.set_ylabel('% Deviation', color='r')
        ax1.plot(ft_x, dev, plotType, color='r')
        ax1.tick_params(axis='y', labelcolor='r')

        ax2 = ax1.twinx()

        ax2.set_ylabel(f'Spectral Density of Clean {entry} Data', color='b')
        ax2.plot(ft_x, ft_yChanged, plotType, color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        fig.tight_layout() 

        plt.title(title)

        if supervised:
            plt.show()
        else:
            plt.savefig(saveLoc + "\\" + title + ".png")
            plt.close()

    def plot_ft_loglog(self, entry: str, fileName: str, supervised=False, saveLoc=None, plotTitle="_FT_LOGLOG_", plotType="-o", sampleSpacing=1, turbulent=False, turbSampleMins=20, gradient=None) -> None:
        """
        Presents a plot of the FFT spectral density with both x and y axes in log10. Turbulent switches on u' = u - u_bar with u_bar evaulated over 
        turbSampleMins minutes averaged over the total time and gradient provides the slope for a line starting at the end of the FT curve.
        Refer to plot_ft_dev for other parameter details.
        """
        if turbulent:
            #Preallocating array of x values
            N = len(self.df.loc[(0 <= self.df['Minute']) & (self.df['Minute'] <= turbSampleMins), entry])
            ft_x = fftfreq(N, sampleSpacing)[:N//2] #The function is symmetric so we are only interested in +ive frequency values

            #Preallocating an array of FFTs over turbSampleMins long snapshots to average over
            snapshots = range(self.df.Minute[len(self.df) - 1]//turbSampleMins - 1)
            ft_y_arr = pd.DataFrame(np.zeros([len(snapshots), N//2]))

            #Going over each turbSampleMins snapshot and FFTing
            for i in snapshots:
                y = self.df.loc[(i*turbSampleMins <= self.df['Minute']) & (self.df['Minute'] <= (i + 1)*turbSampleMins), entry]
                y_bar = y.mean()
                y_turb = y - y_bar

                ft_yTemp = fft(y_turb.values)
                ft_yTemp = 2/N * np.abs(ft_yTemp[:N//2])**2 #Multiplying by 2 to deal with the fact that we chopped out -ive freqs
                ft_y_arr.loc[i] =  ft_yTemp

            #Averaging over all time snapshots
            ft_y = ft_y_arr.mean(axis = 0)

            title = fileName[:len(fileName) - 5] + plotTitle + f"_{turbSampleMins}MIN_" + entry
        
        else:
            N = len(self.df[entry])
            ft_x = fftfreq(N, sampleSpacing)[:N//2] #The function is symmetric so we are only interested in +ive frequency values

            ft_y = fft(self.df[entry].values)
            ft_y = 2/N * np.abs(ft_y[:N//2])**2 #Multiplying by 2 to deal with the fact that we chopped out -ive freqs

            title = fileName[:len(fileName) - 5] + plotTitle + entry
        
        logical1 = ft_y > min(ft_y) #Removing erroneous minimum value which is << the second smallest
        #logical2 = ft_x > ft_x[100] #Arbitrarily cutting out the first 100 elements to get rid of the initial head of the curve
        #ft_x = ft_x[logical1 & logical2]
        #ft_y = ft_y[logical1 & logical2]
        ft_x = ft_x[logical1]
        ft_y = ft_y[logical1]

        plt.loglog(ft_x, ft_y, plotType, color='b')
        plt.xlabel('Frequency Domain')
        plt.ylabel(f'Spectral Density of Clean {entry} Data')
        plt.title(title)

        #Casting a line passing through the final point in the FT with m = gradient
        if gradient is not None:
            #using y = ax^k for loglog plotting
            #yVal = ft_y.loc[len(ft_y) - 10:].mean()
            yVal = ft_y.loc[90:110].mean()
            xVal = ft_x[100]
            c = np.log10(yVal/(xVal**gradient))
            line = (10**c)*ft_x[100:]**gradient #Guarding against div0 errors

            plt.loglog(ft_x[100:], line, color='r')
            plt.legend(['FFT', f'm = {round(gradient, 2)}'])

        if supervised:
            plt.show()
        else:
            plt.savefig(saveLoc + "\\" + title + ".png")
            plt.close()

    def plot_hist(self, entry: str, fileName: str, supervised=False, saveLoc=None, plotTitle="_HIST_", bins=50):
        """
        Plots a histogram of entry with the specified amount of bins. Refer to plot_comparison for other parameters.
        """
        self.df.hist(column=entry, bins=bins)

        title = fileName[:len(fileName) - 5] + plotTitle + entry

        plt.xlabel(entry)
        plt.ylabel('Frequency')
        plt.title(title)

        if supervised:
            plt.show()
        else:
            plt.savefig(saveLoc + "\\" + title + ".png")
            plt.close()

    def std_cutoff(self, entry: str, stdMargain: float) -> pd.Series:
        """
        Returns logicals set to True for data of type entry that lies beyond +-stdMargain standard deviations from the mean of the dataset.
        """
        cutOff = stdMargain*self.std.loc[entry]
        #return self.df.loc[np.abs(self.mean.loc[entry] - self.df[entry]) > cutOff, entry]
        return np.abs(self.mean.loc[entry] - self.df[entry]) > cutOff

    def gradient_cutoff(self, entry: str, diffStdMargain: float) -> pd.Series:
        """
        Returns logicals set to True for type entry data whose gradients lie beyond +-diffStdMargain standard deviations from the mean slope (unit/s).
        """
        #Finding the derivative in unit/s
        dt = self.df.loc[pd.notna(self.df[entry]), 'GlobalSecs'].diff()
        dy = self.df.loc[pd.notna(self.df[entry]), entry].diff()
        #dy/dt
        slopes = dy.div(dt)

        cutOff = diffStdMargain*slopes.std()
        mean = slopes.mean()

        #Finding slopes too steep and returning associated datapoints
        slopeIdx = slopes.index[np.abs(mean - slopes) > cutOff].to_series()
        return self.df.index.isin(slopeIdx)

    def prune_and(self, entry: str, logical1: pd.Series, logical2: pd.Series, iterations=1) -> None:
        """
        Cuts out datapoints of type entry which fit into condition A and condition B as defined by logical1 and logical2 and replaces them
        with linear interpolations. Repeated iterations times.
        """
        for i in range(iterations):
            self.df.loc[(logical1 & logical2), entry] = np.nan

        self.remove_nans(entry, self.df)

    def remove_nans(self, entry: str, df: pd.DataFrame) -> None:
        """
        Cuts out NaNs in entry and removes them with linear interpolations.
        """
        #TODO: This is quite scuffed. Linear interps when two or more neighbouring points are NaN will result in straight lines in that region.
        #Furthermore, for loops are quite cumbersome time-wise, however we tend to only have 40-90 NaNs tops, so it shouldn't affect performance too much.
        for nanIdx in df.index[pd.isna(df[entry])].to_list():
            #If neighbouring points are NaN, recursively find the nearest points which aren't
            xLower, xUpper = self._interp_aux(entry, nanIdx - 1, nanIdx + 1)

            #Finding neighbouring x and y values to interpolate between
            if xLower > 0 and xUpper < len(self.df) - 1:
                xNeighbours = df.GlobalSecs[[xLower, xUpper]].values
                yNeighbours = df.loc[[xLower, xUpper], entry].values

                df.loc[nanIdx, entry] = np.interp(df.GlobalSecs[nanIdx], xNeighbours, yNeighbours) #Linearly interpolating. If we need to extrapolate (i.e. endpoints), we just say that the value = the neighbour

            elif xLower == 0 and xUpper < len(self.df) - 1:
                xNeighbours = df.GlobalSecs[xUpper]
                yNeighbours = df.loc[xUpper, entry]

                df.loc[nanIdx, entry] = np.interp(df.GlobalSecs[nanIdx], xNeighbours, yNeighbours, left=yNeighbours, right=yNeighbours)

            elif xLower > 0 and xUpper == len(self.df) - 1:
                xNeighbours = df.GlobalSecs[xLower]
                yNeighbours = df.loc[xLower, entry]

                df.loc[nanIdx, entry] = np.interp(self.df.GlobalSecs[nanIdx], xNeighbours, yNeighbours, left=yNeighbours, right=yNeighbours)

            else:
                raise ValueError("This dataset is scuffed. Every single point was flagged as bad.")

        #updating mean and std
        self.mean = self.df.mean()
        self.std = self.df.std()

    def _interp_aux(self, entry, left, right) -> tuple[int]:
        """
        Recursively searching for the nearest non-NaN value to interpolate to. left is the index left of the value being interpolated. Vice versa with
        right.
        """
        if pd.isna(self.df.loc[right, entry]) and right < len(self.df) - 1: #Need to check if right isn't already an edgepoint
            return self._interp_aux(entry, left, right + 1)

        elif pd.isna(self.df.loc[left, entry]) and left > 0:
            return self._interp_aux(entry, left - 1, right)
        
        else:
            return (left, right)



    
    # def frequency_cutoff(self, entry: str, freqStdMargain: float, iterations: int) -> None:
    #     """
    #     Slices off data whose frequency lies beyond +-freqStdMargain standard deviations from the frequency. Repeated iterations times. Used to remove bars of data
    #     """
    #     if iterations < 1:
    #         raise ValueError("iterations must be >= 1")

    #     for i in range(iterations):
    #         #Finding the derivative in unit/s
    #         frequencies = self.df.groupby([entry])[entry].count().reset_index(name='Count').sort_values(['Count'], ascending=False)
    #         #print(frequencies)
    #         self.df.hist(column=entry,bins=30)
    #         #print(len(self.df.loc[self.df[entry] == 34.29]))

    #         cutOff = freqStdMargain*frequencies.std()
    #         mean = frequencies.mean()

    #         #Finding slopes too steep and removing associated datapoints
    #         """
    #         slopeIdx = slopes.loc[np.abs(mean - slopes) > cutOff].index
    #         self.df.loc[self.df.index.isin(slopeIdx), entry] = np.nan

    #         #Updating mean and std
    #         self.mean = self.df.mean()
    #         self.std = self.df.std()
    #         """
    