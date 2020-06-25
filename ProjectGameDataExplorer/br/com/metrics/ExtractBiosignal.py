'''
Created on 3 de dez de 2019

@author: eltonss
'''

import sys

from pyphysio import EvenlySignal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphysio as ph
import pyphysio.estimators.Estimators as est
import pyphysio.filters.Filters as flt
import math
from sklearn.preprocessing import Imputer
# import the Signal classes
# import data from included examples
# create a signal
# create a Filter
class FeaturesExtractBiosignal(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.path = "/media/elton/Backup/ColetaDados/Validacao/"

    def processingMetrics(self):
        
        # Loading Dataset
        self.dataset = pd.read_csv("/home/elton/Desktop/Dataset/dataset.csv", sep=',') 
        #dataset_validation = [30,31,32,33,34,35,36,37,41,42,43,44,45,46,47,48,49,50,51]
        dataset_train = [3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 37, 43, 45, 47]
        Players_test =  [31,32,33,34,35,36,41,42,44,48,49,50,51]
        #Tem Player 2
        participant = 46
        sessions = [1,3,4]
        for session  in sessions:
            filter = ((self.dataset['Player'] == participant) & (self.dataset['Session'] == session));
            time_initial_session = np.min(self.dataset[filter]['Interval Initial'])
            time_final_session = np.max(self.dataset[filter]['Interval Final'])
            #if(math.isnan(auxi)):
            #    continue
            df_NegativeEmotion = self.processingMetricFaceEmotion(time_initial_session, time_final_session, participant, session, 'NegativeEmotion')
            df_PostiveEmotion = self.processingMetricFaceEmotion(time_initial_session, time_final_session, participant, session, 'PositiveEmotion')

            df_BVP = self.processingMetricBVP(time_initial_session, time_final_session,participant, session)
            #df_BVP = self.runImputer(df_BVP)         
            
            df_EDA = self.processingMetricEDA(time_initial_session, time_final_session, participant, session)
            #print(df_EDA[:18])
            #df_EDA = self.runImputer(df_EDA)      
            
            df = df_NegativeEmotion.join(df_PostiveEmotion).join(df_EDA).join((df_BVP))
            df = self.runImputer(df)
            url = '/home/elton/Desktop/Dataset/Resultados/MetricsP{}_S{}.csv'.format(participant, session)
           
            df.to_csv(url)
            
    def runImputer(self,DF):
        
        fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=0)
        fill_NaN.fit(DF)
        imputed_DF = pd.DataFrame(fill_NaN.fit_transform(DF).round(2))
        imputed_DF.columns = DF.columns
        imputed_DF.index = DF.index
        return imputed_DF;
    def processingMetricFaceEmotion(self, auxi, auxf, participant, session, typeEmotion):
           
        # try:
            self.PATH_EMOTION_FACE = '/home/elton/Desktop/Dataset/MetricsEmotion.csv'
            
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
            
            #data_emotion.plot()   
            #plt.title("Emotion of the Player {} in Session {} ".format(participant, session))
            #plt.savefig('/home/elton/Pictures/Resultados/Emotion/{}{}_S{}.png'.format(typeEmotion, participant, session), dpi=600)
            #plt.show()
            #plt.close()
            
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
            self.PATH_BVP =  self.path+'Participante {}/BVP.csv'.format(participant)
            
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
            
#             ax1 = plt.subplot(211)
#             ibi.plot()   
#             plt.subplot(212)
#             bvp.plot()
#             plt.grid(b=None)
#             plt.show()
#             plt.savefig('/home/elton/Pictures/Resultados/BVP/BVP{}_S{}.png'.format(participant, session), dpi=600)
#             plt.close()
            
            # id_bad_ibi = ph.BeatOutliers(cache=3, sensitivity=0.25)(ibi)
            # ibi = ph.FixIBI(id_bad_ibi)(ibi)
            fixed_length = ph.FixedSegments(step=10, width=10)       
            TD_HRV_ind, col_names = ph.fmap(fixed_length, ph.preset_hrv_td(), ibi)
            TD_HRV_ind_df = pd.DataFrame(TD_HRV_ind, columns=col_names)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('begin', 1)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('end', 1)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('label', 1)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('IBI_DFA1', 1)
            TD_HRV_ind_df = TD_HRV_ind_df.drop('IBI_DFA2', 1)
        
            col_names = col_names.tolist()
            col_names.remove('begin')
            col_names.remove('end')
            col_names.remove('label')
            col_names.remove('IBI_DFA1')
            col_names.remove('IBI_DFA2')
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
        
        self.PATH_EDA = self.path+'Participante {}/EDA.csv'.format(participant)
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
            
#             ax1 = plt.subplot(211)
#             eda.plot()   
#             plt.grid(b=None)
#             plt.subplot(212)             
#             phasic.plot()            
#             plt.show()
#             plt.savefig('/home/elton/Pictures/Resultados/EDA/edaP{}_S{}.png'.format(participant, session), dpi=600)
#             plt.close()
#             pd.options.display.float_format = '{:.2f}'.format
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
