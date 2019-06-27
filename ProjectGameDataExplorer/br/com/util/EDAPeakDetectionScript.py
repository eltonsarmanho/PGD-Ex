import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
from br.com.util.UnixTime import UnixTime
from datetime import datetime

class EDAPeakDetectionScript:
    global SAMPLE_RATE
    SAMPLE_RATE = 8
    def __init__(self): 
        

        pass;
    
    def datetime_to_float(self,d):
        epoch = datetime.datetime.utcfromtimestamp(0)
        total_seconds =  (d - epoch).total_seconds()
        # total_seconds will be in decimals (millisecond precision)
        return total_seconds
    
    def processEDA(self,signal,startTime,endTime):
        thresh = 0.02;
        offset = 1;
        start_WT = 4;
        end_WT = 4;
        
        
        data = self.loadData_E4(signal)

        df = self.calcPeakFeatures(data,offset,thresh,start_WT,end_WT)
        peakData = df[(df.index >= startTime) & (df.index <= endTime)]
        ts = []
        peak = []
        raw_eda = []
        filtered_eda = []
        amp = []
        for item in peakData.index:
            #print(type(item))
            ts.append(item.to_pydatetime().timestamp())
        index = 0;
        
        for p in peakData['peaks']:
            if(p == 1):
                peak.append(index)
                amp.append(peakData['amp'][index])
            filtered_eda.append(peakData['filtered_eda'][index])
            raw_eda.append(peakData['EDA'][index])
            index = index + 1;
            
        return ts,raw_eda,filtered_eda,peak,amp;
    def findPeaks(self,data, offset, start_WT, end_WT, thres=0, sampleRate=SAMPLE_RATE):
        '''
            This function finds the peaks of an EDA signal and returns basic properties.
            Also, peak_end is assumed to be no later than the start of the next peak. (Is this okay??)
    
            ********* INPUTS **********
            data:        DataFrame with EDA as one of the columns and indexed by a datetimeIndex
            offset:      the number of rising samples and falling samples after a peak needed to be counted as a peak
            start_WT:    maximum number of seconds before the apex of a peak that is the "start" of the peak
            end_WT:      maximum number of seconds after the apex of a peak that is the "rec.t/2" of the peak, 50% of amp
            thres:       the minimum uS change required to register as a peak, defaults as 0 (i.e. all peaks count)
            sampleRate:  number of samples per second, default=8
    
            ********* OUTPUTS **********
            peaks:               list of binary, 1 if apex of SCR
            peak_start:          list of binary, 1 if start of SCR
            peak_start_times:    list of strings, if this index is the apex of an SCR, it contains datetime of start of peak
            peak_end:            list of binary, 1 if rec.t/2 of SCR
            peak_end_times:      list of strings, if this index is the apex of an SCR, it contains datetime of rec.t/2
            amplitude:           list of floats,  value of EDA at apex - value of EDA at start
            max_deriv:           list of floats, max derivative within 1 second of apex of SCR
    
        '''
        EDA_deriv = data['filtered_eda'][1:].values - data['filtered_eda'][:-1].values
        peaks = np.zeros(len(EDA_deriv))
        peak_sign = np.sign(EDA_deriv)
        for i in range(int(offset), int(len(EDA_deriv) - offset)):
            if peak_sign[i] == 1 and peak_sign[i + 1] < 1:
                peaks[i] = 1
                for j in range(1, int(offset)):
                    if peak_sign[i - j] < 1 or peak_sign[i + j] > -1:
                        #if peak_sign[i-j]==-1 or peak_sign[i+j]==1:
                        peaks[i] = 0
                        break
    
        # Finding start of peaks
        peak_start = np.zeros(len(EDA_deriv))
        peak_start_times = [''] * len(data)
        max_deriv = np.zeros(len(data))
        rise_time = np.zeros(len(data))
    
        for i in range(0, len(peaks)):
            if peaks[i] == 1:
                temp_start = max(0, i - sampleRate)
                max_deriv[i] = max(EDA_deriv[temp_start:i])
                start_deriv = .01 * max_deriv[i]
    
                found = False
                find_start = i
                # has to peak within start_WT seconds
                while found == False and find_start > (i - start_WT * sampleRate):
                    if EDA_deriv[find_start] < start_deriv:
                        found = True
                        peak_start[find_start] = 1
                        peak_start_times[i] = data.index[find_start]
                        rise_time[i] = self.get_seconds_and_microseconds(data.index[i] - pd.to_datetime(peak_start_times[i]))
    
                    find_start = find_start - 1
    
                # If we didn't find a start
                if found == False:
                    peak_start[i - start_WT * sampleRate] = 1
                    peak_start_times[i] = data.index[i - start_WT * sampleRate]
                    rise_time[i] = start_WT
    
                # Check if amplitude is too small
                if thres > 0 and (data['EDA'].iloc[i] - data['EDA'][peak_start_times[i]]) < thres:
                    peaks[i] = 0
                    peak_start[i] = 0
                    peak_start_times[i] = ''
                    max_deriv[i] = 0
                    rise_time[i] = 0
    
        # Finding the end of the peak, amplitude of peak
        peak_end = np.zeros(len(data))
        peak_end_times = [''] * len(data)
        amplitude = np.zeros(len(data))
        decay_time = np.zeros(len(data))
        half_rise = [''] * len(data)
        SCR_width = np.zeros(len(data))
    
        for i in range(0, len(peaks)):
            if peaks[i] == 1:
                peak_amp = data['EDA'].iloc[i]
                start_amp = data['EDA'][peak_start_times[i]]
                amplitude[i] = peak_amp - start_amp
    
                half_amp = amplitude[i] * .5 + start_amp
    
                found = False
                find_end = i
                # has to decay within end_WT seconds
                while found == False and find_end < (i + end_WT * sampleRate) and find_end < len(peaks):
                    if data['EDA'].iloc[find_end] < half_amp:
                        found = True
                        peak_end[find_end] = 1
                        peak_end_times[i] = data.index[find_end]
                        decay_time[i] = self.get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[i])
    
                        # Find width
                        find_rise = i
                        found_rise = False
                        while found_rise == False:
                            if data['EDA'].iloc[find_rise] < half_amp:
                                found_rise = True
                                half_rise[i] = data.index[find_rise]
                                SCR_width[i] = self.get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[find_rise])
                            find_rise = find_rise - 1
    
                    elif peak_start[find_end] == 1:
                        found = True
                        peak_end[find_end] = 1
                        peak_end_times[i] = data.index[find_end]
                    find_end = find_end + 1
    
                # If we didn't find an end
                if found == False:
                    min_index = np.argmin(data['EDA'].iloc[i:(i + end_WT * sampleRate)].tolist())
                    peak_end[i + min_index] = 1
                    peak_end_times[i] = data.index[i + min_index]
    
        peaks = np.concatenate((peaks, np.array([0])))
        peak_start = np.concatenate((peak_start, np.array([0])))
        max_deriv = max_deriv * sampleRate  # now in change in amplitude over change in time form (uS/second)
    
        return peaks, peak_start, peak_start_times, peak_end, peak_end_times, amplitude, max_deriv, rise_time, decay_time, SCR_width, half_rise

    def get_seconds_and_microseconds(self,pandas_time):
        return pandas_time.seconds + pandas_time.microseconds * 1e-6

    def calcPeakFeatures(self,data,offset,thresh,start_WT,end_WT):
        returnedPeakData = self.findPeaks(data, offset*SAMPLE_RATE, start_WT, end_WT, thresh, SAMPLE_RATE)
        data['peaks'] = returnedPeakData[0]
        data['peak_start'] = returnedPeakData[1]
        data['peak_end'] = returnedPeakData[3]
    
        data['peak_start_times'] = returnedPeakData[2]
        data['peak_end_times'] = returnedPeakData[4]
        data['half_rise'] = returnedPeakData[10]
        # Note: If an SCR doesn't decrease to 50% of amplitude, then the peak_end = min(the next peak's start, 15 seconds after peak)
        data['amp'] = returnedPeakData[5]
        data['max_deriv'] = returnedPeakData[6]
        data['rise_time'] = returnedPeakData[7]
        data['decay_time'] = returnedPeakData[8]
        data['SCR_width'] = returnedPeakData[9]
    
        featureData = data[data.peaks==1][['EDA','rise_time','max_deriv','amp','decay_time','SCR_width']]
    
        # Replace 0s with NaN, this is where the 50% of the peak was not found, too close to the next peak
        featureData[['SCR_width','decay_time']]=featureData[['SCR_width','decay_time']].replace(0, np.nan)
        featureData['AUC']=featureData['amp']*featureData['SCR_width']
    
        
    
        return data
    

# draws a graph of the data with the peaks marked on it
# assumes that 'data' dataframe already contains the 'peaks' column
    def plotPeaks(self,data, x_seconds, sampleRate = SAMPLE_RATE):
        if x_seconds:
            time_m = np.arange(0,len(data))/float(sampleRate)
        else:
            time_m = np.arange(0,len(data))/(sampleRate*60.)
    
        data_min = min(data['EDA'])
        data_max = max(data['EDA'])
    
        #Plot the data with the Peaks marked
        plt.figure(1,figsize=(20, 5))
        peak_height = data_max * 1.15
        data['peaks_plot'] = data['peaks'] * peak_height
        
        plt.plot(time_m,data['peaks_plot'],'#4DBD33')
        #plt.plot(time_m,data['EDA'])
        plt.plot(time_m,data['filtered_eda'])
        plt.xlim([0,time_m[-1]])
        y_min = min(0, data_min) - (data_max - data_min) * 0.1
        plt.ylim([min(y_min, data_min),peak_height])
        plt.title('EDA with Peaks marked')
        plt.ylabel('$\mu$S')
        if x_seconds:
            plt.xlabel('Time (s)')
        else:
            plt.xlabel('Time (min)')
    
        plt.show()

    def chooseValueOrDefault(self,str_input, default):
           
        if str_input == "":
            return default
        else:
            return float(str_input)

    def loadSingleFile_E4(self,filepath,list_of_columns, expected_sample_rate,freq):
        # Load data
        data = pd.read_csv(filepath)
        
        # Get the startTime and sample rate
        startTime = UnixTime().run(data.columns.values[0])
        sampleRate = float(data.iloc[0][0])
        data = data[data.index!=0]
        data.index = data.index-1
        
        # Reset the data frame assuming expected_sample_rate
        data.columns = list_of_columns
        if sampleRate != expected_sample_rate:
            print('ERROR, NOT SAMPLED AT {0}HZ. PROBLEMS WILL OCCUR\n'.format(expected_sample_rate))
    
        # Make sure data has a sample rate of 8Hz
        data = self.interpolateDataTo8Hz(data,sampleRate,startTime)
        return data
    
    def loadData_E4(self,filepath):
        # Load EDA data
        eda_data = self.loadSingleFile_E4(filepath,["EDA"],4,"250L")
        # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
        eda_data['filtered_eda'] =  self.butter_lowpass_filter(eda_data['EDA'], 1.0, 8, 6)
    
       
        return eda_data
    
    
    def interpolateDataTo8Hz(self,data,sample_rate,startTime):
        print("sample_rate %s " % (sample_rate))
        if sample_rate<8:
            # Upsample by linear interpolation
            if sample_rate==2:
                data.index = pd.date_range(start=startTime, periods=len(data), freq='500L')
            elif sample_rate==4:
                data.index = pd.date_range(start=startTime, periods=len(data), freq='250L')
            data = data.resample("125L").mean()
        else:
            if sample_rate>8:
                # Downsample
                idx_range = list(range(0,len(data))) # TODO: double check this one
                data = data.iloc[idx_range[0::int(int(sample_rate)/8)]]
            # Set the index to be 8Hz
            data.index = pd.date_range(start=startTime, periods=len(data), freq='125L')
    
        # Interpolate all empty values
        data = self.interpolateEmptyValues(data)
        return data
    
    def interpolateEmptyValues(self,data):
        cols = data.columns.values
        for c in cols:
            data.loc[:, c] = data[c].interpolate()
    
        return data
    
    def butter_lowpass(self,cutoff, fs, order=5):
        # Filtering Helper functions
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(self,data, cutoff, fs, order=5):
        # Filtering Helper functions
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = scisig.lfilter(b, a, data)
        return y



