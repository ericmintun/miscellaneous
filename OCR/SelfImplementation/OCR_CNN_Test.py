import numpy as np
import time
import math
import OCR_Layers as nn
import OCR_IDX as idx


if __name__ == "__main__":

	test = 'conv'
	
	if test == 'conv':
		func = lambda : 1
		#func = (lambda : np.random.uniform(0, 1))
		#testInput = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
		testInput = np.array([[0,1,1,0],[1,1,1,0],[1,0,0,0],[0,0,0,1]])
		conv = nn.Convolutional((1,4,4),1,(3,3),func)
		
		test1 = conv.feedForward(testInput)
		test1b = nn.convolve(testInput, conv.A[0,0])
		#testBack = np.array([[2,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
		#test2 = conv.backPropagate(testBack)

		
		print(testInput)
		print(test1)
		#print(test2)
		print(test1b)
		print(conv.A)
		
		#conv.updateWeights(testBack,0.5)
	
	if test == 'rec':
		testInput = np.array([[-5,3,7,3,-7,-1,7],[5,3,7,-3,-7,1,-7]])
		testBack = np.array([[7,2,-1,4,1,-5,1],[1,1,-1,1,1,-1,1]])
		
		rec = nn.LinRec(7)
		
		test1 = rec.feedForward(testInput)
		test2 = rec.backPropagate(testBack)
		
		print(test1)
		print(test2)
	
	
	if test == 'dropout':
		testInput = np.array([[1,2,3,4,5,6,7,8,9,10],[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]])
		testBack = np.array([[8,3,7,2,8,1,9,7,3,2],[-8,-3,-7,-2,-8,-1,-9,-7,-3,-2]])
		fraction = 0.5
		dropout = nn.Dropout(10,fraction)
		
		testOut = dropout.feedForward(testInput)
		print(testOut)
		test2 = dropout.backPropagate(testBack)
		print(test2)
		
	
	
	
	
	if test == 'maxpool':
		testInput = np.array([[[1,2,3,4],[10,7,2,6],[0,7,2,5],[8,7,1,5]],[[0,7,3,2],[6,8,1,3],[8,1,4,2],[7,8,2,1]]])
		testBack = np.array([[[4,2],[1,10]],[[1,1],[1,1]]])
	
		print(testInput)
	
		pool = nn.MaxPool((2,4,4), 2)
	
		testForward = pool.feedForward(testInput)
	
		testBackward = pool.backPropagate(testBack)
	
		print(testForward)
		print(testBackward)