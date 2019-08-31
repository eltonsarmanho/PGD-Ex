'''
Created on 28 de mar de 2019

@author: eltonss
'''
from datetime import datetime
from datetime import timedelta
class UnixTime(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def run(self,strTime):
        #ts = int("1284101485")
        ts = float(strTime)

        # if you encounter a "year is out of range" error the timestamp
        # may be in milliseconds, try `ts /= 1000` in that case
        #dt = datetime.fromtimestamp(ts).strftime('%H:%M:%S');
        #print(dt)
        #print(datetime.fromtimestamp(ts) + timedelta(seconds=1/4))
        #return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S');
        return datetime.fromtimestamp(ts);
    
    def runGMT(self,strTime):
        #ts = int("1284101485")
        ts = float(strTime)

        # if you encounter a "year is out of range" error the timestamp
        # may be in milliseconds, try `ts /= 1000` in that case
        #dt = datetime.fromtimestamp(ts).strftime('%H:%M:%S');
        #print(dt)
        #print(datetime.fromtimestamp(ts) + timedelta(seconds=1/4))
        return datetime.utcfromtimestamp(ts)
        #return datetime.fromtimestamp(ts);
    
    def time_array(self,strTime,count,samplingRate):
        aux = 0;
        array = []
        dt = self.run(strTime)
        print("Count: %s " % (len(count)/samplingRate))
            #diff = accumulate - startTime;
        for ts in count:
            #print(dt+aux*timedelta(seconds=1/4))
            accumulate = dt+aux*timedelta(seconds=1/samplingRate);
            #st= datetime.fromtimestamp(accumulate.timestamp()).strftime('%H:%M:%S');
            
            array.append(accumulate.timestamp())
            #array.append(str(diff))
            aux = aux + 1;
        #print(datetime.fromtimestamp(array[len(array)-1]))
        return array[len(array)-1],array;
    
    def time_inc(self,strTime,value):
        aux = 0;
        array = []
        dt = self.run(strTime)
        accumulate = dt+timedelta(seconds=int(value/1000));
        return accumulate;
    
    def timeFrom(self,strTime,arraySecond):
        array = []
        dt = self.run(strTime)
        for ts in (arraySecond):
            #print(dt+aux*timedelta(seconds=1/4))
            accumulate = dt+timedelta(seconds=ts);
            #diff = accumulate - startTime;
            array.append(accumulate.timestamp())
            #array.append(str(diff))
            
        return array;
    
    def time_array_segment(self,strTime,count,samplingRate):
        array = []
        dt = self.run(strTime)
            #diff = accumulate - startTime;
        
        aux = 0;
        count_segment = int(len(count)/samplingRate)
        for ts in range(count_segment):
            accumulate = dt+aux*timedelta(seconds=1);
            t = datetime.fromtimestamp(accumulate.timestamp()).strftime('%H:%M:%S');
            #diff = accumulate - startTime;
            #print(accumulate)
            array.append(accumulate.timestamp())
            #array.append(str(diff))
            aux = aux + 1;
        print("Last Time: %s" %(datetime.fromtimestamp(array[len(array)-1])))
        return array[len(array)-1],array;  
    
    def time_1(self,strTime,samplingRate):
        aux = 0;
        array = []
        dt = self.run(strTime)
            #diff = accumulate - startTime;
        for ts in (range(1294)):
            #print(dt+aux*timedelta(seconds=1/4))
            accumulate = dt+aux*timedelta(seconds=1);
           
            array.append(accumulate.timestamp())
            #array.append(str(diff))
            aux = aux + 1;
        diff = accumulate - self.run(strTime);
        return array;    
  
    def diffTimeStamp(self,strTimeVideo, strT2):
        """
        Method that calculates difference between video time and arbitrary time

        Parameters
        ----------
        strTimeVideo: String
           video time.
        strT2: String
            arbitrary time.
        """
        tstamp1 = self.runGMT(strTimeVideo)
        tstamp2 = self.run(strT2)

        if tstamp1 > tstamp2:
            td = tstamp1 - tstamp2
        else:
            td = tstamp2 - tstamp1
        td_seconds = int(round(td.total_seconds()))
        return td_seconds;

if __name__ == '__main__':
    ut = UnixTime();
    #ut.run("1553810577");
    ut.time_1("1553810577",1)
    pass
         
            