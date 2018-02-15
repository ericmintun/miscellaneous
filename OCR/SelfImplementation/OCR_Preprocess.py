import png
import numpy as np
import os
import math
import random

#Stores the training and test data sets as lists of filenames.  Can pick training and test data randomly.
class dataSets:

	#Pulls a number of random elements out of data and returns them as a separate list.
	@staticmethod
	def pullRandomSubset(data, number):
		remainingData = list(data) #Copy input data so we don't modify it when popping elements.
		if number > len(remainingData):
			raise ValueError("The requested number is larger than the size of the data set.")
		pulledData = []
		for i in range(number):
			rindex = np.random.randint(0,len(remainingData))
			pulledData.append(remainingData.pop(rindex))
		return remainingData, pulledData

	def __init__(self, baseDirectory, subfolderPrefix, identifierTokens, testFraction, maxPerToken=0):
		self.baseDirectory = baseDirectory
		self.identifierTokens = identifierTokens
		self.subfolderPrefix = subfolderPrefix
		
		#This is a dictionary of 2-tuples.  The key is the token, and the 2-tuple contains the training data list and the test data list.
		self.data = {token : self.fillDataSet(token, testFraction, maxPerToken) for token in self.identifierTokens}
		
		#This is a repeat of the above data indexed differently, since space is cheap.  This is terrible solution but I'm lazy.
		#This is a list of 2-tuples, where the first is the token and the second is the filename.
		self.trainingDataFlat = []
		self.testDataFlat = []
		for token, tokenData in self.data.items():
			for filename in tokenData[0]:
				self.trainingDataFlat.append((token, filename))
			for filename in tokenData[1]:
				self.testDataFlat.append((token,filename))
		
		#self.numTrained = sum(map(lambda x : len(x[1][0]), self.data.items())) 
		#self.numTested = sum(map(lambda x : len(x[1][1]), self.data.items()))
		self.numTrained = len(self.trainingDataFlat)
		self.numTested = len(self.testDataFlat)
		
	#Builds the directory path
	def directoryName(self, token='##'):
		return self.baseDirectory + "\\" + str(token) + "\\" + self.subfolderPrefix + str(token)	
	
	
	#Gets all relevant filenames, and splits them up by training versus test.
	def fillDataSet(self, token, testFraction, maxPerToken=0):
		directory = self.directoryName(token)
		print("Indexing directory: " + directory)
		tokenData = os.listdir(directory)
		
		#Truncate dataset if necessary.
		if maxPerToken>0 and maxPerToken<len(tokenData):
			junkedData, tokenData  = dataSets.pullRandomSubset(tokenData, maxPerToken)
	
		#Pick out the test data
		testSize = int(len(tokenData)*testFraction)
		tokenTrainingData, tokenTestData = dataSets.pullRandomSubset(tokenData, testSize)
	
		return (tokenTrainingData, tokenTestData)
		
	#A generator for the complete filepath to training data.	
	def trainingList(self, token):
		currentData = self.data[token][0]
		for filename in currentData:
			yield self.directoryName(token) + "\\" + filename
			
	#A generator for the complete filepath to test data.	
	def testList(self, token):
		currentData = self.data[token][1]
		for filename in currentData:
			yield self.directoryName(token) + "\\" + filename	

	#Generates data in batches of the specified size without replacement.  Returns a list of 2-tuples, the first of which is the token and the second is the file path
	#Batch not guaranteed to be of size batchSize, if at the end of the dataset.
	def trainingBatch(self, batchSize):
		dataShuffle = list(self.trainingDataFlat)
		random.shuffle(dataShuffle)
		for i in range(0,len(dataShuffle),batchSize):
			if i < len(dataShuffle):
				yield map(lambda x : (x[0], self.directoryName(x[0]) + "\\" + x[1]), dataShuffle[i:i+batchSize])
			else:
				yield map(lambda x : (x[0], self.directoryName(x[0]) + "\\" + x[1]), dataShuffle[i:])
				
	def testBatch(self, batchSize):
		dataShuffle = list(self.testDataFlat)
		random.shuffle(dataShuffle)
		for i in range(0,len(dataShuffle),batchSize):
			if i < len(dataShuffle):
				yield map(lambda x : (x[0], self.directoryName(x[0]) + "\\" + x[1]), dataShuffle[i:i+batchSize])
			else:
				yield map(lambda x : (x[0], self.directoryName(x[0]) + "\\" + x[1]), dataShuffle[i:])		
			
	def getRandomTrainingPoints(self, batchSize):
		dataShuffle = list(self.trainingDataFlat)
		random.shuffle(dataShuffle)
		return map(lambda x : (x[0], self.directoryName(x[0]) + "\\" + x[1]), dataShuffle[:batchSize])
	
#Loads NIST png files and processes them.
class imageReader:
	
	#Static function.  Takes grayscale PNG data and puts it in a 2D numpy array.
	@staticmethod
	def PNGToNumPy(pngdata):	
		return np.vstack(map(lambda x : np.uint8(x[0::3]), pngdata)) #This takes just the first of RGB, assuming they are all equal (correct for NIST data).


	#Static function.  Loads the PNG data from a file and puts it in a NumPy array.	
	@staticmethod
	def getDataFromFilename(filename):
		r = png.Reader(filename)
		pngdata = r.read()[2]
		return imageReader.PNGToNumPy(pngdata)
	
	#Static function.  Normalize data.  Will be unmagic-numbered later.  Maybe
	@staticmethod
	def normalize(data):
		return data/255
	
	#Static function.  Invert data.
	@staticmethod
	def invert(data):
		return np.max(data)-data
	
	def __init__(self, rawImageSize, croppingMargins, coarseGrainFactor, normalize=True, invert=True):
		self.rawImageSize = rawImageSize
		self.croppingMargins = croppingMargins
		self.factor = coarseGrainFactor
		self.normalize=normalize
		self.invert=invert
		self.imageSize = [int((self.rawImageSize[0] - self.croppingMargins[0] - self.croppingMargins[1])/self.factor), int((self.rawImageSize[1] - self.croppingMargins[2] - self.croppingMargins[3])/self.factor)]

	#Crops outer margins of image (top, bottom, left, right).
	def cropData(self, data):
		croppedData = data[self.croppingMargins[0]:(data.shape[0]-self.croppingMargins[1]), self.croppingMargins[2]:(data.shape[1]-self.croppingMargins[3])]
		return croppedData

	#Performs a block coarse-graining, averaging over squares of pixels of size factor.  Any left-over squares are cropped on the right and bottom.	
	def coarseGrainData(self, data):
		newData = np.zeros((int(data.shape[0]/self.factor),int(data.shape[1]/self.factor)))
		for i in range(self.factor):
			for j in range(self.factor):
				newData = newData + data[i::self.factor,j::self.factor]
	
		return newData/(self.factor**2)
	
	#Loads an image from a file and returns the processed data.
	def processedImage(self, filename):
		data = self.getDataFromFilename(filename)
		data = self.cropData(data)
		if self.normalize == True:
			data = imageReader.normalize(data)
		if self.invert == True:
			data = imageReader.invert(data)
		data = self.coarseGrainData(data)
		return data
		
		
		
