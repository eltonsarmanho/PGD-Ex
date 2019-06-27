import biosppy


class ProcessingData:
    
    def __init__(self):
        pass;
    
    def normalizeEDA(self,myarray):
        
        arrayNormalized = myarray;
        max_value = max(myarray);
        min_value = min(myarray);
        arrayNormalized = (arrayNormalized - min_value) / (max_value - min_value); 
        return arrayNormalized;
    
    def getMetricsEDA(self,signal):
        metric = biosppy.tools.signal_stats(signal)
        return metric;
    
    def ProcessedBVPDataE4(self,sig):
        ts, filtered, onsets, ts_hr, hr = biosppy.bvp.bvp(signal=sig, sampling_rate=64., show=True);       
        return  filtered, ts_hr, hr;