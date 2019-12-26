#    E3 Data processor Allows to slide, graph and process Empatica E3/E4 wristband data
#    Copyright (C) 2015 Darien Miranda <dmirand@cicese.edu.mx>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import datetime
import sys


class E3Data:

	def __init__(self, dataType, startTime, samplingRate, data):
		self.dataType = dataType
		self.startTime = float (startTime)
		self.samplingRate = float (samplingRate)
		self.data = data

	def toString(self, unixTime=True):
		if(unixTime):
			return "Data Type: %s, Start Time:%s, End Time:%s  SamplingRate %s" % (self.dataType, self.startTime, self.getEndTime(), self.samplingRate)
		else:
			_string = "Data Type: %s, Start Time:%s, End Time:%s  SamplingRate %s" % (
					self.dataType, datetime.datetime.fromtimestamp(self.startTime)
					, datetime.datetime.fromtimestamp(float(self.getEndTime())), self.samplingRate)
			return _string

	def getData(self):
		return self.data

	def getEndTime(self):
		_startDateTime = datetime.datetime.fromtimestamp(self.startTime)
		_endDateTime = _startDateTime + datetime.timedelta (seconds=len(self.data) / self.samplingRate)
		return  _endDateTime.strftime("%s")

	def getSlide(self, start, end):
		_slideStartTime = datetime.datetime.fromtimestamp(self.startTime) 
		
		_slideStartTime = _slideStartTime + datetime.timedelta(seconds=start)
		return E3Data(self.dataType,
				_slideStartTime.strftime("%s"), self.samplingRate,
				self.data[start * int (self.samplingRate): end * int (self.samplingRate)])

	def getNormalTime(self):
		return datetime.datetime.fromtimestamp(self.startTime)

	def saveToFile(self, _path):
		with open(_path, "w") as _FILE_OUTPUT:
			_FILE_OUTPUT.write(str(self.startTime) + "\n")
			_FILE_OUTPUT.write(str(self.samplingRate) + "\n")
			for _line in self.data:
				_FILE_OUTPUT.writelines(','.join(str(y) for y in _line) + "\n")

	@staticmethod
	def newE3DataFromFilePath(self, _FILE_INPUT_PATH, _DATA_TYPE):
		with open(_FILE_INPUT_PATH, "r") as _FILE_INPUT:
			_lineNumber = 0
			_samplingRate = -1
			_startTime = ""
			_data = []
			for _line in _FILE_INPUT:
				if (_DATA_TYPE == "TAGS"):
					_dataLine = _line.replace("\n", "").split(",")
					_data.append(_dataLine)
					_startTime = 0
					continue;
				if (_lineNumber == 0):
					_startTime = _line.replace("\n", "").split(",")[0]
				if (_lineNumber == 1):
					if not (_DATA_TYPE == "IBI"):
						_samplingRate = _line.replace("\n", "").split(",")[0]
				if(_lineNumber > 1):
					_dataLine = _line.replace("\n", "").split(",")
					_data.append(_dataLine)
				_lineNumber += 1
			return E3Data(_DATA_TYPE, _startTime, _samplingRate, _data)

