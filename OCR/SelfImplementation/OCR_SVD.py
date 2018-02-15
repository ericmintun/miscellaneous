import png
import numpy as np
import time
import math
import OCR_Preprocess as pre

#Builds the matrix A = YX^-1 and offset b from the image data
def buildMatrices(dataByFile, yData, reader):
	
	totalTrained = dataByFile.numTrained
	
	XLength = reader.imageSize[0]*reader.imageSize[1]
	YLength = len(yData)
	
	XMatrix = np.zeros((XLength,XLength))
	YMatrix = np.zeros((YLength,XLength))
	
	runningXVector = np.zeros(XLength)
	runningYVector = np.zeros(YLength)
	
	
	for token, yVector in yData.items():
		print("Training token: " + token)
		
		counter = 0
		
		for filename in dataByFile.trainingList(token):
			counter = counter + 1
			if(counter % 5000 == 0):
				print("On number " + str(counter))
			
			currentData = (reader.processedImage(filename)).flatten()
			
			runningXVector = runningXVector + currentData
			runningYVector = runningYVector + yVector
			
			XMatrix = XMatrix + np.outer(currentData, currentData)
			YMatrix = YMatrix + np.outer(yVector, currentData)
			
			
	XMatrix = XMatrix - np.outer(runningXVector, runningXVector)
	YMatrix = YMatrix - np.outer(runningYVector, runningXVector)
	
	#Normalize
	XMatrix = XMatrix/(totalTrained**2)
	YMatrix = YMatrix/(totalTrained**2)
	runningXVector = runningXVector/totalTrained
	runningYVector = runningYVector/totalTrained
	
	print("Singular value decomposition...")
	start = time.time()
	U, s, V = np.linalg.svd(XMatrix)
	end = time.time()
	print("Singular value decomposition took: " + str(end - start) + " seconds.")
	
	Vt = np.conjugate(np.transpose(V))
	YVt = np.dot(YMatrix,Vt)
	
	#Find the size of the block of singular values of s, and assures these colummns of YVt are zero too.
	#These columns may be set to any solution, and we will setting them to zero.  Not sure this is the best strategy.
	
	#This assumes s is square and singular values are sorted in descending order.
	svNonZerosNum = np.count_nonzero(s)
	#print("Singular values: ")
	#print(s)
	if np.count_nonzero(YVt.T[svNonZerosNum:]) > 0:
		print("The linear system of equations has no solution.")
		return []
	
	sReduced = s[:svNonZerosNum]
	sInv = sReduced**(-1.0)
	sInvFull = np.append(sInv, np.zeros(s.shape[0]-svNonZerosNum))
	SInv = np.diag(sInvFull)
	
	Ut = np.conjugate(np.transpose(U))
	A = np.dot(YVt, np.dot(SInv, Ut))
	b = -np.dot(A, runningXVector) - runningYVector
	
	return A, b, s

def doMaxIndicesAgree(a,b):
	return np.argmax(a)==np.argmax(b)
	
def percentTestPassed(A, b, dataByFile, yData, reader):
	numToken = len(yData)
	numPassed = 0
	totalTested = dataByFile.numTested
	guessMatrix = {(rowToken, columnToken) : 0 for rowToken in yData.keys() for columnToken in yData.keys()} 
	
	for token, yVector in yData.items():

		print("Testing token number " + token)
			
		counter = 0
		tokenTested = len(dataByFile.data[token][1])
		
		for filename in dataByFile.testList(token):
			counter = counter + 1
			if(counter % 5000 == 0):
				print("On number " + str(counter))
				
			currentData = (reader.processedImage(filename)).flatten()
			
			predictionVec = np.dot(A, currentData)+b
			prediction = np.argmax(predictionVec)
			
			for tok, vec in yData.items():
				if np.argmax(vec) == prediction:
					predictionToken = tok
					break
			
			
			
			guessMatrix[(token,predictionToken)]=guessMatrix[(token,predictionToken)]+1/tokenTested
			
			if prediction==np.argmax(yVector):
				numPassed = numPassed + 1
				
		
	return numPassed/totalTested, guessMatrix
			

#Constructs an independent vector for each token, and builds a dictionary for which the token is the vector's key.
def buildYVectors(identifierTokens):
	yVectors = {}
	for i, token in enumerate(identifierTokens):
		vector = np.zeros(len(identifierTokens))
		vector[i] = 1.0
		yVectors[token] = vector
		
	return yVectors
	
if __name__ == "__main__":
	baseDirectory = "..\\NISTHandwritingDatabase\\by_class"
	subfolderPrefix = "train_"
	typeTokens = ['30','31','32','33','34','35','36','37','38','39']
	testFraction = 0.1
	imageSize = [128,128]
	croppingMargins = [32,32,32,32]
	maxImagesPerChar = 3000
	coarseGrainFactor = 2
	normalize = True
	invert = True
	logFilename = "OCR_SVD_log.txt"
	weightsFilename = "OCR_SVD_Weights.npz"
	
	reader = pre.imageReader(imageSize, croppingMargins, coarseGrainFactor, normalize, invert)
	
	
	timeStart = time.time()

	dataByFile = pre.dataSets(baseDirectory, subfolderPrefix, typeTokens, testFraction, maxImagesPerChar)
	yData = buildYVectors(typeTokens)

	
	fileIndexCheckpoint = time.time()
	fileIndexTime = fileIndexCheckpoint - timeStart
	
	
	A, b, s = buildMatrices(dataByFile, yData, reader)
	
	trainingCheckpoint = time.time()
	trainingTime = trainingCheckpoint - fileIndexCheckpoint
	
	percentPassed, guessMatrix = percentTestPassed(A, b, dataByFile, yData, reader)
	
	timeEnd = time.time()
	testingTime = timeEnd - trainingCheckpoint
	totalTime = timeEnd - timeStart
	
	print("Fraction passed: " + str(percentPassed))
	
	np.savez_compressed(weightsFilename, A=A, b=b, s=s)
	
	f = open(logFilename, 'w')
	f.write("Results for OCR performed using singular value decomposition.\n")
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
	
	
