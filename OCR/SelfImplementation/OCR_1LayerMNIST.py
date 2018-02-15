import numpy as np
import time
import math
import OCR_IDX as idx

#Builds a random prior for A and b by choosing a random order 1 number for each entry.
def randomPrior(xSize, ySize):
	return np.random.randn(ySize, xSize), np.random.randn(ySize)

def softmaxY(A, b, x):
	yGuess = np.dot(A, x) + b
	yGuessExp = np.exp(yGuess)
	Z = np.sum(yGuessExp)
	return yGuessExp/Z
	
def updateMatrices(A, b, batch, stepSize):
	ADiff = np.zeros(A.shape)
	bDiff = np.zeros(b.shape)
		
	for x, y in batch:
			
		yGuess = softmaxY(A, b, x)
			
		#print(yGuess)
		ADiff = ADiff + np.outer(yGuess, x) - np.outer(y, x)
		bDiff = bDiff + yGuess - y
		
	normalization = max(1.0, np.trace(np.dot(ADiff, ADiff.T)) + np.dot(bDiff,bDiff))
		
	ADiff = ADiff/math.sqrt(normalization)
	bDiff = bDiff/math.sqrt(normalization)
	newA = A - ADiff
	newb = b - bDiff

	return newA, newb
	
def yVectorMNIST(label, size):
	vec = np.zeros(size)
	vec[label] = 1
	return vec
	
def buildDataPoint(ySize, imageLabelPair,imageNorm):
	return (imageLabelPair[0].flatten()/imageNorm, yVectorMNIST(imageLabelPair[1], ySize))

def getRandomIndexList(listSize, dataSize):
	indexList = list(range(dataSize))
	randomList = []
	for i in range(listSize):
		j = np.random.randint(len(indexList))
		randomList.append(indexList.pop(j))
		
	return randomList
	
def buildRandomBatch(data, imageNorm, ySize, batchSize):
	batchIndexList = getRandomIndexList(batchSize, len(data))
	batch = map(lambda x : buildDataPoint(ySize, data[x],imageNorm), batchIndexList)
	return batch
	
def train(data, imageNorm, ySize, imageSize, batchSize, stepSize, numSteps):
	xLength = imageSize[0]*imageSize[1]
	
	A, b = randomPrior(xLength, ySize)
	
	for i in range(numSteps):
		if (i+1) % 100 == 0:
			print("On training step " + str(i+1))
		batch = buildRandomBatch(data, imageNorm, ySize, batchSize)
		A, b = updateMatrices(A, b, batch, stepSize)
		
	return A, b
	
def test(A, b, data, imageNorm, ySize):
	totalTested = len(data)
	guessMatrix = np.zeros((ySize,ySize))
	numTokenTested = np.zeros(ySize)
	numPassed = 0
	
	counter = 0
	for rawDataPoint in data:
		counter = counter + 1
		if counter % 1000 == 0:
			print("On test " + str(counter))
		x, y = buildDataPoint(ySize, rawDataPoint, imageNorm)
		yGuess = softmaxY(A, b, x)
		
		actualToken = np.argmax(y)
		guessToken = np.argmax(yGuess)
		
		if actualToken == guessToken:
			numPassed = numPassed + 1
			
		guessMatrix[actualToken,guessToken]=guessMatrix[actualToken,guessToken]+1
		numTokenTested[actualToken] = numTokenTested[actualToken]+1
		
	for i in range(ySize):
		guessMatrix[i] = guessMatrix[i]/numTokenTested[i]
		
	percentPassed = numPassed/totalTested	
		
	return percentPassed, guessMatrix
	
if __name__ == "__main__":
	trainingImageFilename = "..\\MNIST\\train-images.idx3-ubyte"
	trainingLabelFilename = "..\\MNIST\\train-labels.idx1-ubyte"
	testImageFilename = "..\\MNIST\\t10k-images.idx3-ubyte"
	testLabelFilename = "..\\MNIST\\t10k-labels.idx1-ubyte"
	imageSize = [28,28]
	numDigits = 10
	trainingSteps = 1000
	batchSize = 100
	stepSize = 0.5
	imageNorm = 255
	
	logFilename = "OCR_1LayerMNIST_log.txt"
	weightsFilename = "OCR_1LayerMNIST_Weights.npz"
	
	timeStart = time.time()
	
	print("Loading training data from file.")
	trainingData = idx.readMNISTData(trainingImageFilename, trainingLabelFilename)
	print("Loading test data from file.")
	testData = idx.readMNISTData(testImageFilename, testLabelFilename)
	
	timeLoad = time.time()
	
	print("Training.")
	A, b = train(trainingData, imageNorm, numDigits, imageSize, batchSize, stepSize, trainingSteps)
	
	timeTrain = time.time()
	
	print("Testing.")
	percentPassed, guessMatrix = test(A, b, testData, imageNorm, numDigits)
	
	timeEnd = time.time()
	
	print("Fraction passed: " + str(percentPassed))
	
	np.savez_compressed(weightsFilename, A=A, b=b)
	
	f = open(logFilename, 'w')
	f.write("Results for OCR performed using 1-layer softmax and gradient descent.\n")
	f.write("The weights determined by this run have been saved to: " + weightsFilename + "\n")
	f.write("Training data loaded from: " + trainingImageFilename + " and " + trainingLabelFilename + "\n")
	f.write("Test data loaded from: " + testImageFilename + " and " + testLabelFilename + "\n")	
	f.write("Total number of training steps: " + str(trainingSteps) + "\n")
	f.write("Batch size per training step: " + str(batchSize) + "\n")
	f.write("Step size for training: " + str(stepSize) + "\n")
	f.write("Total time: " + str(timeEnd - timeStart) + "\n")
	f.write("File loading time: " + str(timeLoad - timeStart) + "\n")
	f.write("Training time: " + str(timeTrain - timeLoad) + "\n")
	f.write("Testing time: " + str(timeEnd - timeTrain) + "\n")
	f.write("\n")
	f.write("\n")
	f.write("Total fraction of tests passed: " + str(percentPassed)+ "\n")
	f.write("\n")
	f.write("\n")
	f.write("Prediction matrix:\n")
	f.write("Rows indicate actual values, columns indicate predicted values.\n")
	f.write("\n")
	f.write("    |")
	for digit in range(numDigits):
		f.write("   " + "{:2d}".format(digit) + "   ") 
	f.write("\n")
	f.write("----+")
	for digit in range(numDigits):
		f.write("--------") 
	f.write("\n")	
	for rowDigit in range(numDigits):
		f.write(" " + "{:2d}".format(rowDigit) + " |")
		for columnDigit in range(numDigits):
			f.write(" " + "{:6.4f}".format(guessMatrix[rowDigit, columnDigit]) + " ")
		f.write("\n")
	
	
	f.close()	