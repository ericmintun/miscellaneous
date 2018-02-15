import numpy as np
import time
import math
from scipy.signal import convolve2d

#Helper functions
def theta(input):
	return 1 * ( input > 0)
	
def softmax(input):
	return 1/np.sum(np.exp(input[:,np.newaxis] - input), axis=0)


#Abstract class, does nothing
class Layer:
	
	
	def __init__(self):
		self.forwardTime = 0
		self.backwardTime = 0
		self.updateTime =0
			
	def feedForward(self, input):
		return input
		
	def backPropagate(self, input):
		return input
				
	def updateWeights(self, stepSize, beta1, beta2, epsilon):
		return
		
	def calculateDiffs(self, input):
		return (0, 0)
		
	def setArgs(self, args):
		return
		
	def resetTimes(self):
		self.forwardTime = 0
		self.backwardTime = 0
		self.updateTime = 0
		
		
#Linearly rectifies a vector		
class LinRec(Layer):
		
	def __init__(self, size):
		#self.batchSize = size[0]
		self.length = size
		
		super().__init__()
		

				
		
	def feedForward(self, input):
	
		startTime = time.time()
		
		self.X = input.reshape((-1,self.length))
		output = (np.abs(self.X) + self.X)/2.0
		
		endTime = time.time()
		self.forwardTime = self.forwardTime + endTime - startTime
		
		return output
		
	def backPropagate(self, input):
	
		startTime = time.time()
		
		output = input.reshape((-1,self.length)) * theta(self.X)
		
		endTime = time.time()
		self.backwardTime = self.forwardTime + endTime - startTime
	
		return output
		
		
#Drops a random percentage of nodes.  Applies to vectors.
class Dropout(Layer):

	def __init__(self, size, fraction):
		self.length = size
			
		self.fraction = fraction
		
		super().__init__()
		
	def setArgs(self, args):
		self.fraction = args[0]
		
	def feedForward(self, input):
	
		startTime = time.time()
	
		self.dropVec = theta(np.random.uniform(0,1,self.length) - self.fraction)
		output = input.reshape((-1,self.length)) * self.dropVec / (1 - self.fraction)

		endTime = time.time()
		self.forwardTime = self.forwardTime + endTime - startTime		
		
		return output
		
	def backPropagate(self, input):
	
		startTime = time.time()
		
		output = input.reshape((-1,self.length)) * self.dropVec / (1 - self.fraction)
		
		endTime = time.time()
		self.backwardTime = self.forwardTime + endTime - startTime
		
		return output
		
		
#Max pooling.  Takes in a 3D matrix and max pools in the first two dimensions.  Takes the input size as (x,y,z), outputs a 3D matrix of size (x/factor, y/factor, z).
class MaxPool(Layer):

	def __init__(self, size, factor):
		
		self.numNodes = size[0]
		self.imageXSize = size[1]
		self.imageYSize = size[2]
		if size[1] % factor != 0 or size[2] % factor != 0:
			print("Max pooling factor does not evenly divide image size.  Edges will be dropped.")
		self.outputXSize = int(size[1]/factor)
		self.outputYSize = int(size[2]/factor)
							
		self.factor = factor
		
		super().__init__()
		
	def feedForward(self, input):
	
		startTime = time.time()
		
		self.X = input.reshape((-1,self.imageXSize,self.imageYSize))
		Y = np.zeros((len(self.X),self.outputXSize,self.outputYSize))
		self.maxXs = np.zeros((len(self.X),self.outputXSize,self.outputYSize))
		factor = self.factor #Just to get rid of some selfs
		
		for i in range(self.outputXSize):
			for j in range(self.outputYSize):
				maxArgSub = np.argmax(self.X[:,factor*i:factor*(i+1),factor*j:factor*(j+1)].reshape((self.X.shape[0],factor*factor)), axis=1)
				maxArgFlat = (i*factor + np.floor(maxArgSub / factor)) * self.imageXSize + (j*factor + maxArgSub % factor)
				#maxArg = np.array(np.unravel_index(maxArgFlat, (self.imageXSize, self.imageYSize))).T
				#print(maxArg)
				Y[:,i,j] = np.array([self.X.reshape((-1,self.imageXSize*self.imageYSize))[i,int(maxArg)] for i, maxArg in enumerate(maxArgFlat)]) * self.factor**2
				self.maxXs[:,i,j] = maxArgFlat
					
		output = Y.reshape((-1,self.numNodes,self.outputXSize,self.outputYSize))
		
		endTime = time.time()
		self.forwardTime = self.forwardTime + endTime - startTime			
					
		return output
		
	def backPropagate(self, input):

		startTime = time.time()
	
		diffsIn = input.reshape((-1,self.outputXSize,self.outputYSize))
		diffsOut = np.zeros((len(diffsIn),self.imageXSize*self.imageYSize))
		for n in range(len(diffsIn)):
			for i in range(self.outputXSize):
				for j in range(self.outputYSize):
					diffsOut[n,int(self.maxXs[n,i,j])] = diffsIn[n,i,j]
					
		output = diffsOut.reshape((-1,self.numNodes,self.imageXSize,self.imageYSize)) * self.factor**2
			
		endTime = time.time()
		self.backwardTime = self.forwardTime + endTime - startTime			
					
		return output
		
		
class Linear(Layer):

	def __init__(self, size, priorFunction, *priorFunctionArgs):
	
		self.inputSize = size[0]
		self.outputSize = size[1]
		
		self.A = np.array([ priorFunction(*priorFunctionArgs) for i in range(self.inputSize*self.outputSize)]).reshape((self.outputSize, self.inputSize))
		self.b = np.array([ priorFunction(*priorFunctionArgs) for i in range(self.outputSize) ])
		
		self.ADiffsRunning = np.zeros((self.outputSize, self.inputSize))
		self.bDiffsRunning = np.zeros(self.outputSize)
		
		self.ADiffsSqdRunning = np.zeros((self.outputSize, self.inputSize))
		self.bDiffsSqdRunning = np.zeros(self.outputSize)
		
		super().__init__()
		
	def feedForward(self, input):
	
		startTime = time.time()
	
		self.X = input.reshape((-1,self.inputSize))
		output = np.einsum('ij,aj',self.A, self.X) + self.b
		
		endTime = time.time()
		self.forwardTime = self.forwardTime + endTime - startTime	
		
		return output
		
	def backPropagate(self, input):
	
		startTime = time.time()	
	
	
		YDiffs = input.reshape((-1,self.outputSize))
		#print(YDiffs[0,:])
		output = np.einsum('aj,jk',YDiffs, self.A)
		#print(output[0,:])
		
		endTime = time.time()
		self.backwardTime = self.forwardTime + endTime - startTime	
		
		return output
		
	def updateWeights(self, stepSize, beta1, beta2, epsilon):
	
		self.ADiffsRunning = self.ADiffsRunning * beta1 + (1-beta1) * self.ADiff
		self.bDiffsRunning = self.bDiffsRunning * beta1 + (1-beta1) * self.bDiff
		
		self.ADiffsSqdRunning = self.ADiffsSqdRunning * beta2 + (1-beta2) * self.ADiff * self.ADiff
		self.bDiffsSqdRunning = self.bDiffsSqdRunning * beta2 + (1-beta2) * self.bDiff * self.bDiff
	
		self.A = self.A - stepSize * self.ADiffsRunning / (np.sqrt(self.ADiffsSqdRunning) + epsilon)
		self.b = self.b - stepSize * self.bDiffsRunning / (np.sqrt(self.bDiffsSqdRunning) + epsilon)
		
		return	
		
	def calculateDiffs(self, input):
	
		startTime = time.time()	
	
		YDiffs = input.reshape((-1,self.outputSize))
		self.ADiff = np.sum((self.X.T[:,np.newaxis,:]*YDiffs.T).T,axis=0)
		self.bDiff = np.sum(YDiffs,axis=0)
		
		l1Sum = np.sum(self.ADiff) + np.sum(self.bDiff)
		l2Sum = np.sum(self.ADiff * self.ADiff) + np.dot(self.bDiff,self.bDiff)
		
		#ADiff = ADiff/math.sqrt(normalization)
		#bDiff = bDiff/math.sqrt(normalization)
		
		#self.A = self.A - ADiff * stepSize
		#self.b = self.b - bDiff * stepSize
		
		endTime = time.time()
		self.updateTime = self.forwardTime + endTime - startTime
		
		return l1Sum, l2Sum


def convolve(image, window):
	windowXSize = window.shape[0]
	windowYSize = window.shape[1]
	imageXSize = image.shape[0]
	imageYSize =  image.shape[1]
	
	windowXCenter = math.floor(windowXSize / 2)
	windowYCenter = math.floor(windowYSize / 2)
	
	output = np.zeros((imageXSize, imageYSize))
	
	for i in range(windowXSize):
	
		if i < windowXCenter:
			outputXStart = windowXCenter - i
			outputXEnd = imageXSize
			imageXStart = 0
			imageXEnd = imageXSize - (windowXCenter - i)
		else:
			outputXStart = 0
			outputXEnd = imageXSize - (i - windowXCenter)
			imageXStart = i - windowXCenter
			imageXEnd = imageXSize
	
		for j in range(windowYSize):
		
			if j < windowYCenter:
				outputYStart = windowYCenter - j
				outputYEnd = imageYSize
				imageYStart = 0
				imageYEnd = imageYSize - (windowYCenter - j)
			else:
				outputYStart = 0
				outputYEnd = imageYSize - (j - windowYCenter)
				imageYStart = j - windowYCenter
				imageYEnd = imageYSize
		
			output[outputXStart:outputXEnd,outputYStart:outputYEnd] = output[outputXStart:outputXEnd,outputYStart:outputYEnd] + window[-(i+1),-(j+1)] * image[imageXStart:imageXEnd,imageYStart:imageYEnd]
			
	return output
			
		
class Convolutional(Layer):

	def __init__(self, inputSize, nodesOut, windowSize, priorFunction, *priorFunctionArgs):
		
		self.numNodesIn = inputSize[0]
		self.inputXSize = inputSize[1]
		self.inputYSize = inputSize[2]
		
		self.numNodesOut = nodesOut
		self.windowXSize = windowSize[0]
		self.windowYSize = windowSize[1]
		
		self.A = np.array([ priorFunction(*priorFunctionArgs) for i in range(self.numNodesIn*self.numNodesOut*self.windowXSize*self.windowYSize)]).reshape((self.numNodesIn,self.numNodesOut,self.windowXSize,self.windowYSize))
		self.b = np.array([ priorFunction(*priorFunctionArgs) for i in range(self.numNodesOut) ])
		
		self.ADiffsRunning = np.zeros((self.numNodesIn,self.numNodesOut,self.windowXSize,self.windowYSize))
		self.bDiffsRunning = np.zeros(self.numNodesOut)		
		
		self.ADiffsSqdRunning = np.zeros((self.numNodesIn,self.numNodesOut,self.windowXSize,self.windowYSize))
		self.bDiffsSqdRunning = np.zeros(self.numNodesOut)		
		
		super().__init__()
		
		
	def feedForward(self, input):
	
		startTime = time.time()	
	
		self.X = input.reshape((-1, self.numNodesIn, self.inputXSize, self.inputYSize))
		output = np.zeros((len(self.X), self.numNodesOut, self.inputXSize, self.inputYSize))
		batchSize = len(self.X)
		
		for n in range(batchSize):
			for j in range(self.numNodesOut):
				for i in range(self.numNodesIn):
					output[n,j] = output[n,j]+convolve2d(self.X[n,i], self.A[i,j],mode='same')
					#output[n,j] = output[n,j]+convolve(self.X[n,i], self.A[i,j])
				#output[n,j] = output[n,j] + self.b[j]
				
		#print(self.X[0,0])
		#print(np.max(self.A))
		#print(np.max(self.X))
		#print(self.b)
		#print(np.sum(output, axis=(0,2,3))/(len(self.X)*self.inputXSize* self.inputYSize))
					
		endTime = time.time()
		self.forwardTime = self.forwardTime + endTime - startTime		

					
		return output
		
	def backPropagate(self, input):

		startTime = time.time()
	
		YDiff = input.reshape((-1, self.numNodesOut, self.inputXSize, self.inputYSize))
		output = np.zeros((len(YDiff), self.numNodesIn, self.inputXSize, self.inputYSize))
		batchSize = len(output)

		for n in range(batchSize):
			for j in range(self.numNodesIn):
				for i in range(self.numNodesOut):
					output[n,j] = output[n,j]+convolve2d(YDiff[n,i], np.fliplr(np.flipud(self.A[j,i])),mode='same')
					#output[n,j] = output[n,j]+convolve(YDiff[n,i], np.fliplr(np.flipud(self.A[j,i])))
					
		endTime = time.time()
		self.backwardTime = self.forwardTime + endTime - startTime				
					
		return output
		
	def updateWeights(self, stepSize, beta1, beta2, epsilon):
	
		self.ADiffsRunning = self.ADiffsRunning * beta1 + (1-beta1) * self.ADiff
		self.bDiffsRunning = self.bDiffsRunning * beta1 + (1-beta1) * self.bDiff
		
		self.ADiffsSqdRunning = self.ADiffsSqdRunning * beta2 + (1-beta2) * self.ADiff * self.ADiff
		self.bDiffsSqdRunning = self.bDiffsSqdRunning * beta2 + (1-beta2) * self.bDiff * self.bDiff
	
		self.A = self.A - stepSize * self.ADiffsRunning / (np.sqrt(self.ADiffsSqdRunning) + epsilon)
		self.b = self.b - stepSize * self.bDiffsRunning / (np.sqrt(self.bDiffsSqdRunning) + epsilon)
		
		return	
		
	def calculateDiffs(self, input):
	
		startTime = time.time()	
	
		YDiff = input.reshape((-1, self.numNodesOut, self.inputXSize, self.inputYSize))
		self.ADiff = np.zeros((self.numNodesIn,self.numNodesOut,self.windowXSize,self.windowYSize))
		padX = math.floor(self.windowXSize/2)
		padY = math.floor(self.windowYSize/2)
		
		for m in range(self.windowXSize):
				
			if m <= padX:
				XXStart = padX-m
				XXEnd = self.X.shape[2]
				YXStart = 0
				YXEnd = YDiff.shape[2]-(padX-m)
			else:
				XXStart = 0
				XXEnd = self.X.shape[2] - (m-padX)
				YXStart = (m-padX)
				YXEnd = YDiff.shape[2]
				
			for n in range(self.windowYSize):
				#An elegant but way too slow solution.
				#paddedX =  np.pad(self.X[:,i,:,:],((0,0),(m,2*padX-m),(n,2*padY-n)),'constant')
				#paddedDiff = np.pad(YDiff[:,j,:,:],((0,0),(padX,padX),(padY,padY)),'constant')
				#ADiff[i,j,m,n] = np.sum(paddedX * paddedDiff)				
							
				if n <= padY:
					XYStart = padY-n
					XYEnd = self.X.shape[3]
					YYStart = 0
					YYEnd = YDiff.shape[3]-(padY-n)
				else:
					XYStart = 0
					XYEnd = self.X.shape[3] - (n-padY)
					YYStart = (n-padY)
					YYEnd = YDiff.shape[3]	
						
				#for i in range(self.numNodesIn):
				#	for j in range(self.numNodesOut):
				#		ADiff[i,j,m,n] = np.sum(self.X[:,i,XXStart:XXEnd,XYStart:XYEnd]*YDiff[:,j,YXStart:YXEnd,YYStart:YYEnd])
				self.ADiff[:,:,m,n] = np.sum(  self.X[:,:,np.newaxis,XXStart:XXEnd,XYStart:XYEnd].transpose((0,3,4,1,2))*YDiff[:,:,np.newaxis,YXStart:YXEnd,YYStart:YYEnd].transpose((0,3,4,2,1)), axis=(0,1,2)     )
						
		self.bDiff = np.array([np.sum(index) for index in YDiff.transpose((1,0,2,3))])
		#print(bDiff)
		#print(np.max(YDiff[:,0,:,:]))
		
			
		l1Sum = np.sum(self.ADiff) + np.sum(self.bDiff)
		l2Sum = np.sum(self.ADiff * self.ADiff) + np.sum(self.bDiff*self.bDiff)
		
		#ADiff = ADiff/math.sqrt(normalization)
		#bDiff = bDiff/math.sqrt(normalization)
		
		#self.A = self.A - ADiff * stepSize
		#self.b = self.b - bDiff * stepSize
		#print(self.b)
		
		endTime = time.time()
		self.updateTime = self.forwardTime + endTime - startTime	
		
		return (l1Sum, l2Sum)
		
#Final training layer.  Flips from feeding forward to back propagating in a single step.
class SoftmaxRelativeEntropy:
	
	def __init__(self,numClasses):
		self.numClasses = numClasses
		self.reverseTime = 0
		
	def resetTimes(self):
		self.reverseTime = 0
		
	def reversePropagation(self, input, labels):
	
		startTime = time.time()	
	
		output = np.array([softmax(dataPoint) for dataPoint in input.reshape((-1,self.numClasses))]) - labels
	
		endTime = time.time()
		self.reverseTime = self.reverseTime + endTime - startTime	
	
		return output
		
	def loss(self, input, labels):
	
		return (0 - np.sum(labels * np.log(np.array([softmax(dataPoint) for dataPoint in input.reshape((-1,self.numClasses))]))))
		
	def finishTest(self, input):
		return np.argmax(input)


class NeuralNet:

	def __init__(self, stepSize, beta1, beta2, epsilon):
		self.layers = []
		self.outputLayer = None
		
		self.runningNorm = 0
		self.timeStep = 0
		self.stepSize = stepSize
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		
	def appendHLayer(self, newLayer):
		self.layers.append(newLayer)
		
	def setArgs(self, argsList):
		for layer, args in zip(self.layers, argsList):
			layer.setArgs(args)
			
	def printTimes(self):
		print("Current times spent per layer:")
		for i, layer in enumerate(self.layers):
			print("Layer " + str(i+1) + ":")
			print("Feed forward time: " + "{:.2f}".format(layer.forwardTime))
			print("Back propagation time: " + "{:.2f}".format(layer.backwardTime))
			print("Update time: " + "{:.2f}".format(layer.updateTime)) 
			print("Total time: " + "{:.2f}".format(layer.forwardTime +  layer.backwardTime + layer.updateTime))
		print("Output layer time: " + "{:.2f}".format(self.outputLayer.reverseTime))

	def resetTimes():
		for layer in self.layers:
			layer.resetTimes()
		self.outputLayer.resetTimes()
		
	def train(self, data):
		self.timeStep = self.timeStep + 1
		forwardsData = np.array([dataPoint[0] for dataPoint in data])
		labels = np.array([dataPoint[1] for dataPoint in data])
		l2SumRunning = 0
		l1SumRunning = 0
		for layer in self.layers:
			forwardsData = layer.feedForward(forwardsData)
			
		backwardsData = self.outputLayer.reversePropagation(forwardsData, labels)
		loss = self.outputLayer.loss(forwardsData, labels)
		
		for layer in reversed(self.layers):
			l1Sum, l2Sum = layer.calculateDiffs(backwardsData)
			backwardsData = layer.backPropagate(backwardsData)
			l2SumRunning = l2SumRunning + l2Sum
			l1SumRunning = l1SumRunning + l1Sum
		
		stepSizeMod = self.stepSize * np.sqrt(1-self.beta2**self.timeStep)/(1-self.beta1**self.timeStep)
		epsilonMod = self.epsilon * np.sqrt(1-self.beta2**self.timeStep)
		#print(l2SumRunning)
		#self.runningNorm = self.runningNorm * self.beta2 + (1-self.beta2) * l2SumRunning
		#print(stepSizeMod)
		#print(epsilonMod)
		#print(self.runningNorm)
		#norm = stepSizeMod / (np.sqrt(self.runningNorm) + epsilonMod)
		#print(norm)
		
		#preNorm = np.sqrt(preNorm)
		#norm = stepSize/max(300.0,preNorm)	
			
		for layer in self.layers:
			layer.updateWeights(stepSizeMod, self.beta1, self.beta2, epsilonMod)
			
		return (loss, stepSizeMod)
		
	def infer(self, data):
		forwardsData = data
		for layer in self.layers:
			forwardsData = layer.feedForward(forwardsData)
		result = self.outputLayer.finishTest(forwardsData)
		return result
		
		
		
		