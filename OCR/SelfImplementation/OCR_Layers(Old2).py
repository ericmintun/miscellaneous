import numpy as np
import time
import math
import scipy.sparse as sparse

#Abstract class, does nothing
class Layer:
	
	self.qDiagonal = False
	
	def __init__(self):
		
		
	def feedForward(self, input):
		return np.array([])
		
	def derivatives(self):
		return np.array([])
	
	def derivative(self, index):
		return
			
		
	def updateWeights(self, diffs):
		return
		
		
#Linearly rectifies a vector		
class LinRec(Layer):
	
	def rectify(input):
		if input > 0:
			return input
		else:
			return 0
			
	def theta(input):
		return 1 * ( input > 0)
	
	def __init__(self, size):
		self.batchSize = size[0]
		self.length = size[1]
			
		self.X = np.zeros(size)
		
		
	def feedForward(input):
		self.X = input
		data = self.X.reshape(self.batchSize*self.length)
		rectifyVec = np.vectorize(data)
		return rectifyVec(data).reshape((self.batchSize, self.length))
		
	def derivatives(self):		
		return [np.diag(theta(x) for x in self.X)]
		
	def derivative(self, indices):
		if indices[1] != indices[2]:
			return 0
		else:
			return theta(self.X[(indices[0],indices[1])])
		
#Drops a random percentage of nodes.  Applies to vectors.
class Dropout(Layer):

	def __init__(self, size, fraction):
		self.batchSize = size[0]
		self.length = size[1]
			
		self.X = np.zeros(size)
		
		self.fraction = fraction
		
	def feedForward(self, input):
		self.X = input
		self.dropVec = 1*(np.random.uniform(0,1,self.length) < self.fraction)
		return self.X * self.dropVec
		
	def derivatives(self):
		return [np.diag(self.dropVec) for x in self.X]
	
	def derivative(self, indices):
		if indices[1] != indices[2]:
			return 0
		else:
			return self.dropVec[indices[1]]
		
#Max pooling.  Takes in a 3D matrix and max pools in the first two dimensions.  Takes the input size as (x,y,z), outputs a 3D matrix of size (x/factor, y/factor, z).
class MaxPool(Layer):

	def __init__(self, size, factor):
		
		self.batchSize = size[0]
		self.numNodes = size[1]
		self.imageXSize = size[2]
		self.imageYSize = size[3]
		if size[2] % factor != 0 or size[3] % factor != 0:
			print("Max pooling factor does not evenly divide image size.  Edges will be dropped.")
		self.outputXSize = size[2]
		self.outputYSize = size[3]
		
		self.inputShape = (self.batchSize, self.numNodes, self.imageXSize, self.imageYSize)
		self.outputShape = (self.batchSize, self.numNodes, self.outputXSize, self.outputYSize)
		
		self.X = np.zeros(self.inputShape)
		
		self.maxXs = np.zeros((self.batchSize,self.numNodes,self.imageXSize,self.imageYSize))
		
		self.factor = factor
		
	def feedForward(self, input):
		self.X = input
		Y = np.zeros(self.outputShape)
		
		self.X.reshape((self.batchSize*self.numNodes,self.imageXSize,self.imageYSize))
		Y.reshape((self.batchSize*self.numNodes,self.outputXSize,self.outputYSize))
		self.maxXs.reshape((self.batchSize*self.numNodes,self.imageXSize,self.imageYSize))
		
		for i in range(self.outputXSize):
			for j in range(self.outputYSize):
				maxArgFlat = i*self.outputXSize + j + np.argmax(self.X[:,factor*i:factor*(i+1),factor*j:factor*j:factor*(j+1)].reshape((self.X.shape[0],factor*factor)), axis=1)
				maxArg = np.array(np.unravel_index(maxArgFlat, (self.imageXSize, self.imageYSize))).T
				Y[:,i,j] = [self.X[:,index[0],index[1]] for index in maxArg]
				for n in range(len(maxArg)):
					self.maxXs[n,maxArg[n,0],maxArg[n,1]] = 1

		self.X.reshape((self.batchSize,self.numNodes,self.imageXSize,self.imageYSize))
		Y.reshape((self.batchSize,self.numNodes,self.outputXSize,self.outputYSize))
		self.maxXs.reshape((self.batchSize,self.numNodes,self.imageXSize,self.imageYSize))	
					
		return Y
		
	def derivatives(self):
		self.derivatives = np.zeros((self.batchSize,self.numNodes,self.imageXSize,self.imageYSize,self.outputXSize,self.outputYSize))
		return self.derivatives
	
	def derivative(self, index):
		return self.derivatives[index]
		
class Linear(Layer):

	def __init__(self, size, priorFunction, *priorFunctionArgs):
	
		self.inputSize = size[0]
		self.outputSize = size[1]
		
		self.X = np.zeros(self.inputSize)
		self.Y = np.zeros(self.outputSize)
		
		self.A = np.array([ priorFunction(priorFunctionArgs) for i in range(self.inputSize) for j in range(self.outputSize)])
		self.b = np.array([ priorFunction(priorFunctionArgs) for i in range(self.outputSize) ])
		
	def feedForward(self, input):
		self.Y = np.dot(self.A, self.X) + self.b
		return self.Y
		
	def derivatives(self):
		return self.A
		
	def derivative(self, index):
		return self.A[index]
		
	def updateWeights(self, diffs):
		
		