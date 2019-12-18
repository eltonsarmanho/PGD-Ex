'''
Created on 3 de dez de 2019

@author: eltonss
'''


import sys


from pyphysio import EvenlySignal

from ProjectGameDataExplorer.br.com.util import UnixTime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphysio as ph
import pyphysio.estimators.Estimators as est
import pyphysio.filters.Filters as flt


# import the Signal classes
# import data from included examples
# create a signal
# create a Filter
class FeaturesExtractBiosignal(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def processingMetrics(self):
        
        # Loading Dataset
        self.dataset = pd.read_csv("/home/eltonss/Desktop/dataset.csv", sep=',') 
        P = 47
        array = [1,2,3,4]
        _UT = UnixTime.UnixTime()
        for session  in array:
            filter = ((self.dataset['Player'] == P) & 
                          (self.dataset['Session'] == session));
            auxi = np.min(self.dataset[filter]['Interval Initial'])
            auxf = np.max(self.dataset[filter]['Interval Final'])
                
            df_NegativeEmotion = self.processingMetricFaceEmotion(auxi, auxf, P, session,'NegativeEmotion')
            df_PostiveEmotion = self.processingMetricFaceEmotion(auxi, auxf, P, session,'PositiveEmotion')

            df_BVP = self.processingMetricBVP(auxi, auxf, P, session)          
            df_EDA = self.processingMetricEDA(auxi, auxf, P, session)
            df = df_NegativeEmotion.join(df_PostiveEmotion).join(df_EDA).join((df_BVP))
            url = '/home/eltonss/Desktop/Resultados/MetricsP{}_S{}.csv'.format(P, session)
            df.to_csv(url)

    def processingMetricFaceEmotion(self, auxi, auxf, participant, session,typeEmotion):
           
        #try:
            self.PATH_EMOTION_FACE = '/home/eltonss/Desktop/MetricsEmotion.csv'
            
            data = pd.read_csv(self.PATH_EMOTION_FACE) 
            data = data[(data['Session'] == session) & (data['Player'] == participant) & 
                        ((data['Time'] >= auxi) & (data['Time'] <= auxf))]
            startTime = data['Time'].iloc[0]
            sampleRate = 30
            #data = data['NegativeEmotion']
            data_emotion = EvenlySignal(values=data[typeEmotion],
                   sampling_freq=sampleRate,
                   signal_type=typeEmotion,
                   start_time=startTime)
           
            print("Duration Emotion (Session %s) %s: %s: " %(session,typeEmotion,data_emotion.get_duration()))
            print("Start Time: %s" % (startTime))
            #print("End Time: %s " % (data['Time'][len(data['Time'])-1]))
            #print("End Time: %s " % (auxf)) 
            # Filtering
            data_emotion = flt.Normalize('maxmin')(data_emotion)
            # Select area of interest
            
            data_emotion.plot()   
            plt.title("Emotion of the Player {} in Session {} ".format(participant, session))
            plt.savefig('/home/eltonss/Pictures/Resultados/Emotion/{}{}_S{}.png'.format(typeEmotion,participant, session), dpi=600)
            #plt.show()
            plt.close()
            
            #,,ph.Min,ph.StDev,ph.Range,ph.Median
            fixed_length = ph.FixedSegments(step=10, width=10)  
             
            data_emotion_ind, col_names = ph.fmap(fixed_length,ph.preset_activity(prefix='{}_'.format(typeEmotion)),data_emotion)
            # col_names[2] = 'LabelBVP'
            data_emotion_ind_df = pd.DataFrame(data_emotion_ind, columns=col_names)
            data_emotion_ind_df=data_emotion_ind_df[['label','begin','end','{}_maximum'.format(typeEmotion),'{}_minimum'.format(typeEmotion),
                                                     '{}_mean'.format(typeEmotion),'{}_range'.format(typeEmotion),'{}_sd'.format(typeEmotion)]]
            #print("Begin %s " % (data_emotion_ind_df['begin']))
            #print("End %s " % (data_emotion_ind_df['end']))
            if(typeEmotion == 'PositiveEmotion'):
                data_emotion_ind_df = data_emotion_ind_df.drop('begin', 1)
                data_emotion_ind_df = data_emotion_ind_df.drop('end', 1)
            data_emotion_ind_df = data_emotion_ind_df.drop('label', 1)
            #url = '/home/eltonss/Desktop/Resultados/EmotionMetricsP{}_S{}.csv'.format(participant,session)
            #data_emotion_ind_df.to_csv(url)
            return data_emotion_ind_df;
           
        #except:
        #     print("Oops!", sys.exc_info()[0], "occured.")
        #     print("Erro in processingMetricEmotion")
          
    def processingMetricBVP(self, auxi, auxf, participant, session):
        
        try:
        # loading BVP
            self.PATH_BVP = '/media/eltonss/9A2494A8249488C1/Users/eltonn/Dropbox/Dados Experimentos/Experimentos/Dados Coletados/Participante {}/BVP.csv'.format(participant)
            data = pd.read_csv(self.PATH_BVP)            
            startTime = float(data.columns.values[0])
            sampleRate = float(data.iloc[0][0])
            data = data[data.index != 0]
            data.index = data.index - 1 
            data = [row[0] for row in data.get_values()]
            bvp = EvenlySignal(values=data,
                   sampling_freq=sampleRate,
                   signal_type='bvp',
                   start_time=startTime) 
            
            # Filtering            
            bvp = bvp.resample(fout=bvp.get_sampling_freq() * 2, kind='cubic')
            bvp = flt.Normalize('maxmin')(bvp)
            # Select area of interest             
            bvp = bvp.segment_time(auxi, auxf)
            print("Duration of BVP (Session %s): %s" % (session,bvp.get_duration()))
            ibi = est.BeatFromBP()(bvp)
            
            ax1 = plt.subplot(211)
            ibi.plot()   
            plt.subplot(212)
            bvp.plot()
            plt.savefig('/home/eltonss/Pictures/Resultados/BVP/BVP{}_S{}.png'.format(participant, session), dpi=600)
            plt.close()
            
            # id_bad_ibi = ph.BeatOutliers(cache=3, sensitivity=0.25)(ibi)
            # ibi = ph.FixIBI(id_bad_ibi)(ibi)
            fixed_length = ph.FixedSegments(step=10, width=10)       
            TD_HRV_ind, col_names = ph.fmap(fixed_length, ph.preset_hrv_td(), ibi)
            # col_names[2] = 'LabelBVP'
            TD_HRV_ind_df = pd.DataFrame(TD_HRV_ind, columns=col_names)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('begin', 1)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('end', 1)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('label', 1)
            #url = '/home/eltonss/Desktop/Resultados/MetricsBVP.csv'
            #TD_HRV_ind_df.to_csv(url)
            return TD_HRV_ind_df;
            # url = '/home/eltonss/Desktop/Resultados/BVP/BVPmetricsP{}_S{}.csv'.format(participant,session)
            # TD_HRV_ind_df.to_csv(url)
        except:
            print("Oops!", sys.exc_info()[0], "occured.")
            print("Erro in processingMetricBVP")
        
    def processingMetricEDA(self, auxi, auxf, participant, session):
        print("processingMetricEDA")     
        self.PATH_EDA= '/media/eltonss/9A2494A8249488C1/Users/eltonn/Dropbox/Dados Experimentos/Experimentos/Dados Coletados/Participante {}/EDA.csv'.format(participant)
        try:
                # loading EDA
            data = pd.read_csv(self.PATH_EDA)            

            startTime = float(data.columns.values[0])
            sampleRate = float(data.iloc[0][0])
            data = data[data.index != 0]
            data.index = data.index - 1
            data = [row[0] for row in data.get_values()]
            

            eda = EvenlySignal(values=data,
                   sampling_freq=sampleRate,
                   signal_type='EDA',
                   start_time=startTime)
            
            
            # filtering and normalization : 
            # remove high frequency noise
            # resampling : decrease the sampling frequency by cubic interpolation
            eda = eda.resample(fout=8, kind='cubic')
            filter = ph.ConvolutionalFilter(irftype='gauss',
                                            win_len=8 / eda.get_sampling_freq())
            eda = filter(eda)
            normalize = flt.Normalize('maxmin')
            eda = normalize(eda)
            
            # Select area of interest                    
            eda = eda.segment_time(auxi, auxf)
            print("Duration of EDA (Session %s): %s" % (session,eda.get_duration()))
            # create estimator
            driver = ph.DriverEstim()(eda)                  
            
            phasic, tonic, _ = ph.PhasicEstim(delta=0.02)(driver)
            ax1 = plt.subplot(211)
            eda.plot()   
            plt.subplot(212)             
            phasic.plot()
            plt.savefig('/home/eltonss/Pictures/Resultados/EDA/edaP{}_S{}.png'.format(participant, session), dpi=600)
            plt.close()
            
            # fixed length windowing
            fixed_length = ph.FixedSegments(step=10, width=10)       
            # we use the preset indicators for the phasic signal.
            # We need to define the minimum amplitude of the peaks that will be considered
            PHA_ind, col_names = ph.fmap(fixed_length, ph.preset_phasic(delta=0.02), phasic)
            PHA_ind_df = pd.DataFrame(PHA_ind, columns=col_names)
      
            PHA_ind_df = PHA_ind_df.drop('begin', 1)
            PHA_ind_df = PHA_ind_df.drop('end', 1)
            #PHA_ind_df = PHA_ind_df.shift(1, axis=1)
            PHA_ind_df = PHA_ind_df.drop('label', 1)
            #url = '/home/eltonss/Desktop/Resultados/MetricsEDA.csv'
            #PHA_ind_df.to_csv(url)
            return PHA_ind_df
        except:
            print("Oops!", sys.exc_info()[0], "occured.")
            print("Erro in processingMetricEDA")
             
if __name__ == '__main__':
    run = FeaturesExtractBiosignal()
    run.processingMetrics()