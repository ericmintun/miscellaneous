import png
import numpy as np
import time
import math
import OCR_Preprocess as pre
import OCR_SVD as svd #Used since the testing algorithm is identical.
#import OCR_IDX as idx

#Builds a random prior for A and b by choosing a random order 1 number for each entry.
def randomPrior(xSize, ySize):
	return np.random.randn(ySize, xSize), np.random.randn(ySize)


def buildMatrices(dataByFile, yData, reader, batchSize, stepSize, numSteps):
	
	XLength = reader.imageSize[0]*reader.imageSize[1]
	YLength = len(yData)
	
	A, b = randomPrior(XLength, YLength)
	
	counter = 0
	
	#for batch in dataByFile.trainingBatch(batchSize):
	for i in range(numSteps):
		batch = dataByFile.getRandomTrainingPoints(batchSize)
	
		counter = counter + 1
		if(counter % 1 == 0):
			print("On batch number " + str(counter))
		
		ADiff = np.zeros(A.shape)
		bDiff = np.zeros(b.shape)
		
		for token, filename in batch:
			
			currentXData = (reader.processedImage(filename)).flatten()
			currentYData = yData[token]
			
			yGuess = np.dot(A, currentXData) + b
			yGuessExp = np.exp(yGuess)
			Z = np.sum(yGuessExp)
			
			#print(yGuess)
			ADiff = ADiff + np.outer(currentYData, currentXData) - np.outer(yGuessExp, currentXData)/Z
			bDiff = bDiff + currentYData - yGuessExp/Z
		
		normalization = max(1.0, np.trace(np.dot(ADiff, ADiff.T)) + np.dot(bDiff,bDiff))
		
		ADiff = ADiff/math.sqrt(normalization)
		bDiff = bDiff/math.sqrt(normalization)
		
		A = A + ADiff*stepSize
		b = b + bDiff*stepSize
		
	return A,b 
	
if __name__ == "__main__":
	baseDirectory = "..\\NISTHandwritingDatabase\\by_class"
	subfolderPrefix = "train_"
	typeTokens = ['30','31','32','33','34','35','36','37','38','39']
	testFraction = 0.1
	imageSize = [128,128]
	croppingMargins = [32,32,32,32]
	maxImagesPerChar = 0
	numSteps = 1000
	batchSize = 100
	stepSize = 0.5
	coarseGrainFactor = 2
	normalize = True
	invert = True
	logFilename = "OCR_1Layer_log.txt"
	weightsFilename = "OCR_1Layer_Weights.npz"
	
	reader = pre.imageReader(imageSize, croppingMargins, coarseGrainFactor, normalize, invert)
	
	
	timeStart = time.time()

	dataByFile = pre.dataSets(baseDirectory, subfolderPrefix, typeTokens, testFraction, maxImagesPerChar)
	yData = svd.buildYVectors(typeTokens)

	
	fileIndexCheckpoint = time.time()
	fileIndexTime = fileIndexCheckpoint - timeStart
	
	
	A, b = buildMatrices(dataByFile, yData, reader, batchSize, stepSize, numSteps)
	
	trainingCheckpoint = time.time()
	trainingTime = trainingCheckpoint - fileIndexCheckpoint
	
	percentPassed, guessMatrix = svd.percentTestPassed(A, b, dataByFile, yData, reader)
	
	timeEnd = time.time()
	testingTime = timeEnd - trainingCheckpoint
	totalTime = timeEnd - timeStart
	
	print("Fraction passed: " + str(percentPassed))
	
	np.savez_compressed(weightsFilename, A=A, b=b)
	
	f = open(logFilename, 'w')
	f.write("Results for OCR performed using 1-layer softmax and gradient descent.\n")
	f.write("The weights determined by this run have been saved to: " + weightsFilename + "\n")
	f.write("Files loaded from: " + dataByFile.directoryName('##') + "\n")
	f.write("Character tokens used (## in the filename): " + str(typeTokens)+"\n")
	f.write("Margin cropping in pixels: Top: " + str(croppingMargins[0]) + " Bottom: " + str(croppingMargins[1]) + " Left: " + str(croppingMargins[2]) + " Right: " + str(croppingMargins[3])+ "\n")
	f.write("Coarse graining factor: " + str(coarseGrainFactor)+ "\n")
	f.write("Fraction of images randomly chosen for testing: " + str(testFraction)+ "\n")
	f.write("Max number of each character considered (randomly chosen): " + (str(maxImagesPerChar)+ "\n" if maxImagesPerChar > 0 else "None\n"))
	f.write("\n")
	f.write("Total number of images trained: " + str(dataByFile.numTrained)+ "\n")
	f.write("Total number of images tested: " + str(dataByFile.numTested)+ "\n")
	f.write("Time indexing files: " + str(fileIndexTime) + "\n")
	f.write("Time training: " + str(trainingTime) + "\n")
	f.write("Time testing: " + str(testingTime) + "\n")
	f.write("Total time: " + str(totalTime) + "\n")
	f.write("\n")
	f.write("\n")
	f.write("Total fraction of tests passed: " + str(percentPassed)+ "\n")
	f.write("\n")
	f.write("\n")
	f.write("This is the prediction matrix.  Provides the fraction of the time a given character was guessed to be a given other character.\n")
	f.write("Rows indicate actual values, columns the predicted values.\n")
	f.write("\n")
	f.write("    |")
	for token in typeTokens:
		f.write("   " + str(token) + "   ") 
	f.write("\n")
	f.write("----+")
	for token in typeTokens:
		f.write("--------") 
	f.write("\n")	
	for rowToken in typeTokens:
		f.write(" " + str(rowToken) + " |")
		for columnToken in typeTokens:
			f.write(" " + "{:6.4f}".format(guessMatrix[(rowToken, columnToken)]) + " ")
		f.write("\n")
	
	
	f.close()	
	