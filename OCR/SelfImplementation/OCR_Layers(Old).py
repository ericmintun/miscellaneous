import numpy as np
import time
import math
import scipy.sparse as sparse

def vecCheck(size):
	if type(size)==tuple:
		if len(size) > 1:
			print("Layer init given a size corresponding to a multidimensional array but it only accepts a vector.  Only the first dimension will be used.")
			return size[0]
		elif type(size)==int:
			return size
		else:
			print("Layer init given invalid size.  Getting set to zero.")
			return 0

#Abstract class, does nothing
class Layer:
	
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
		self.length = vecCheck(size)
			
		self.X = np.zeros(self.length)
		self.Y = np.zeros(self.length)
		
		
	def feedForward(input):
		self.X = input
		rectifyVec = np.vectorize(self.rectify)
		self.Y = rectifyVec(self.X)
		
		return self.Y
		
	def derivatives(self):		
		return np.diag(theta(self.X))
		
	def derivative(self, indices):
		if indices[0] != indices[1]:
			return 0
		else:
			return theta(self.X[indices[0]])
		
#Drops a random percentage of nodes.  Applies to vectors.
class Dropout(Layer):

	def __init__(self, size, fraction):
		self.length = vecCheck(size)
			
		self.X = np.zeros(self.length)
		self.Y = np.zeros(self.length)
		
		self.fraction = fraction
		
	def feedForward(self, input):
		self.X = input
		self.dropVec = 1*(np.random.uniform(0,1,self.length) < self.fraction)
		self.Y = self.X * self.dropVec
		return np.array([])
		
	def derivatives(self):
		return np.diag(self.dropVec)
	
	def derivative(self, indices):
		if indices[0] != indices[1]:
			return 0
		else:
			return self.dropVec[indices[0]]
		
#Max pooling.  Takes in a 3D matrix and max pools in the first two dimensions.  Takes the input size as (x,y,z), outputs a 3D matrix of size (x/factor, y/factor, z).
class MaxPool(Layer):

	def __init__(self, size, factor):
		
		if size[0] % factor != 0 or size[1] % factor != 0:
			print("Max pooling factor does not evenly divide image size.  Edges will be dropped.")
		self.outputSize = (math.floor(size[0]/factor),math.floor(size[1]/factor),size[3])
		self.inputSize = size
		
		self.X = np.zeros(self.inputSize)
		self.Y = np.zeros(self.outputSize)
		
		self.derivatives = np.zeros(tuple(list(self.inputSize),list(self.outputSize)))
		
		self.factor = factor
		
	def feedForward(self, input):
		self.X = input
		
		for i in range(self.outputSize[0]):
			for j in range(self.outputSize[1]):
				maxArgFlat = i*self.outputSize[0] + j + np.argmax(self.X[factor*i:factor*(i+1),factor*j:factor*j:factor*(j+1)], axis=2)
				maxArg = np.unravel_index(maxArgFlat, self.X.shape)
				maxArgReshaped = np.array(maxArg).T
				self.Y[i,j] = [self.X[tuple(index)] for index in maxArgReshaped]
				for n in range(len(maxArgReshaped)):
					self.derivatives[tuple(i,j,n,maxArgReshaped[n,0],maxArgReshaped[n,1],n] = 1

		return self.Y
		
	def derivatives(self):
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
		
		