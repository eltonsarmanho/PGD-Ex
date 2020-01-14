'''
Created on 3 de dez de 2019

@author: eltonss
'''

import sys

from pyphysio import EvenlySignal

from br.com.util.UnixTime import UnixTime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphysio as ph
import pyphysio.estimators.Estimators as est
import pyphysio.filters.Filters as flt
import math

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
        self.dataset = pd.read_csv("/home/elton/Desktop/datasetValidacao.csv", sep=',') 
        P = [30,31,32,33,34,35,36,37,41,42,43,44,45,46,47,48,49,50,51]
        P = 30
        array = [1,2,3, 4]
        _UT = UnixTime()
        for session  in array:
            filter = ((self.dataset['Player'] == P) & 
                          (self.dataset['Session'] == session));
            auxi = float(np.min(self.dataset[filter]['Interval Initial']))
            auxf = float(np.max(self.dataset[filter]['Interval Final']))
            if(math.isnan(auxi)):
                continue
            df_NegativeEmotion = self.processingMetricFaceEmotion(auxi, auxf, P, session, 'NegativeEmotion')
            df_PostiveEmotion = self.processingMetricFaceEmotion(auxi, auxf, P, session, 'PositiveEmotion')

            df_BVP = self.processingMetricBVP(auxi, auxf, P, session)          
            df_EDA = self.processingMetricEDA(auxi, auxf, P, session)
            df = df_NegativeEmotion.join(df_PostiveEmotion).join(df_EDA).join((df_BVP))
            url = '/home/elton/Desktop/Resultados/MetricsP{}_S{}.csv'.format(P, session)
           
            df.to_csv(url)

    def processingMetricFaceEmotion(self, auxi, auxf, participant, session, typeEmotion):
           
        # try:
            self.PATH_EMOTION_FACE = '/home/elton/Desktop/MetricsEmotion.csv'
            
            data = pd.read_csv(self.PATH_EMOTION_FACE) 
            data = data[(data['Session'] == session) & (data['Player'] == participant) & 
                        ((data['Time'] >= float(auxi)) & (data['Time'] <= float(auxf)))]
            print(data['Time'].iloc[-1])
            startTime = data['Time'].iloc[0]
            sampleRate = 30
            data_emotion = EvenlySignal(values=data[typeEmotion],
                   sampling_freq=sampleRate,
                   signal_type=typeEmotion,
                   start_time=startTime)
           
            print("Duration Emotion (Session %s) %s: %s: " % (session, typeEmotion, data_emotion.get_duration()))
            print("Start Time: %s" % (startTime))
            print("End Time: %s" % (data_emotion.get_end_time()))
            # print("End Time: %s " % (data['Time'][len(data['Time'])-1]))
            # print("End Time: %s " % (auxf)) 
            # Filtering
            data_emotion = flt.Normalize('maxmin')(data_emotion)
            # Select area of interest
            
            data_emotion.plot()   
            plt.title("Emotion of the Player {} in Session {} ".format(participant, session))
            plt.savefig('/home/elton/Pictures/Resultados/Emotion/{}{}_S{}.png'.format(typeEmotion, participant, session), dpi=600)
            plt.show()
            plt.close()
            
            # ,,ph.Min,ph.StDev,ph.Range,ph.Median
            fixed_length = ph.FixedSegments(step=10, width=10)  
             
            data_emotion_ind, col_names = ph.fmap(fixed_length, ph.preset_activity(prefix='{}_'.format(typeEmotion)), data_emotion)
            # col_names[2] = 'LabelBVP'
            data_emotion_ind_df = pd.DataFrame(data_emotion_ind, columns=col_names)
            data_emotion_ind_df = data_emotion_ind_df[['label', 'begin', 'end', '{}_maximum'.format(typeEmotion), '{}_minimum'.format(typeEmotion),
                                                     '{}_mean'.format(typeEmotion), '{}_range'.format(typeEmotion), '{}_sd'.format(typeEmotion)]]
            col_names = [ '{}_maximum'.format(typeEmotion), '{}_minimum'.format(typeEmotion),
                          '{}_mean'.format(typeEmotion), '{}_range'.format(typeEmotion), '{}_sd'.format(typeEmotion)] 
            # print("Begin %s " % (data_emotion_ind_df['begin']))
            # print("End %s " % (data_emotion_ind_df['end']))
            if(typeEmotion == 'PositiveEmotion'):
                data_emotion_ind_df = data_emotion_ind_df.drop('begin', 1)
                data_emotion_ind_df = data_emotion_ind_df.drop('end', 1)
            data_emotion_ind_df = data_emotion_ind_df.drop('label', 1)
            
           
            for col in col_names :
                aux = []
                for value in data_emotion_ind_df[col]:
                    aux.append(np.round(float(value),2))
                data_emotion_ind_df[col] = aux;
         
            return data_emotion_ind_df;

          
    def processingMetricBVP(self, auxi, auxf, participant, session):
        
        try:
        # loading BVP
            self.PATH_BVP = '/home/elton/Documents/Experimento/Validacao/Participante {}/BVP.csv'.format(participant)
            data = pd.read_csv(self.PATH_BVP)            
            startTime = float(data.columns.values[0])
            
            sampleRate = float(data.iloc[0][0])
            data = data[data.index != 0]
            data.index = data.index - 1 
            data = [row[0] for row in data.values]
            bvp = EvenlySignal(values=data,
                   sampling_freq=sampleRate,
                   signal_type='bvp',
                   start_time=startTime) 
            # Filtering            
            bvp = bvp.resample(fout=bvp.get_sampling_freq() * 2, kind='cubic')

            bvp = flt.Normalize('maxmin')(bvp)
            # Select area of interest             
            bvp = bvp.segment_time(auxi, auxf)
            print("Duration of BVP (Session %s): %s" % (session, bvp.get_duration()))
            print("startTime: %s" % (startTime))
            print("sampleRate: %s" % (sampleRate))

            ibi = est.BeatFromBP()(bvp)
            
            ax1 = plt.subplot(211)
            ibi.plot()   
            plt.subplot(212)
            bvp.plot()
            plt.grid(b=None)
            plt.show()
            plt.savefig('/home/elton/Pictures/Resultados/BVP/BVP{}_S{}.png'.format(participant, session), dpi=600)
            plt.close()
            
            # id_bad_ibi = ph.BeatOutliers(cache=3, sensitivity=0.25)(ibi)
            # ibi = ph.FixIBI(id_bad_ibi)(ibi)
            fixed_length = ph.FixedSegments(step=10, width=10)       
            TD_HRV_ind, col_names = ph.fmap(fixed_length, ph.preset_hrv_td(), ibi)
            TD_HRV_ind_df = pd.DataFrame(TD_HRV_ind, columns=col_names)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('begin', 1)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('end', 1)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('label', 1)
        
            col_names = col_names.tolist()
            col_names.remove('begin')
            col_names.remove('end')
            col_names.remove('label')
            for col in col_names :
                aux = []
                for value in TD_HRV_ind_df[col]:
                    aux.append(np.round(float(value),2))
                TD_HRV_ind_df[col] = aux;
            return TD_HRV_ind_df
         
        except:
            print("Oops!", sys.exc_info()[0], "occured.")
            print("Erro in processingMetricBVP")
        
    def processingMetricEDA(self, auxi, auxf, participant, session):
        print("processingMetricEDA")     
        self.PATH_EDA = '/home/elton/Documents/Experimento/Validacao/Participante {}/EDA.csv'.format(participant)
        try:
                # loading EDA
            data = pd.read_csv(self.PATH_EDA)            

            startTime = float(data.columns.values[0])
            sampleRate = float(data.iloc[0][0])
            data = data[data.index != 0]
            data.index = data.index - 1
            data = [row[0] for row in data.values]

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
            print("Duration of EDA (Session %s): %s" % (session, eda.get_duration()))
            # create estimator
            driver = ph.DriverEstim()(eda)                  
            
            phasic, tonic, _ = ph.PhasicEstim(delta=0.02)(driver)
            ax1 = plt.subplot(211)
            eda.plot()   
            plt.grid(b=None)
            plt.subplot(212)             
            phasic.plot()            
            plt.show()
            plt.savefig('/home/elton/Pictures/Resultados/EDA/edaP{}_S{}.png'.format(participant, session), dpi=600)
            plt.close()
            pd.options.display.float_format = '{:.2f}'.format
            # fixed length windowing
            fixed_length = ph.FixedSegments(step=10, width=10)       
            # we use the preset indicators for the phasic signal.
            # We need to define the minimum amplitude of the peaks that will be considered
            PHA_ind, col_names = ph.fmap(fixed_length, ph.preset_phasic(delta=0.02), phasic)
            PHA_ind_df = pd.DataFrame(PHA_ind, columns=col_names)
      
            PHA_ind_df = PHA_ind_df.drop('begin', 1)
            PHA_ind_df = PHA_ind_df.drop('end', 1)
            # PHA_ind_df = PHA_ind_df.shift(1, axis=1)
            PHA_ind_df = PHA_ind_df.drop('label', 1)
            # url = '/home/eltonss/Desktop/Resultados/MetricsEDA.csv'
            # PHA_ind_df.to_csv(url)
            col_names = col_names.tolist()
            col_names.remove('begin')
            col_names.remove('end')
            col_names.remove('label')
            for col in col_names :
                aux = []
                for value in PHA_ind_df[col]:
                    aux.append(np.round(float(value),2))
                PHA_ind_df[col] = aux;
            

            return PHA_ind_df
        except:
            print("Oops!", sys.exc_info()[0], "occured.")
            print("Erro in processingMetricEDA")

             
if __name__ == '__main__':
    run = FeaturesExtractBiosignal()
    run.processingMetrics()
