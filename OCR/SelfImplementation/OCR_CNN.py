import numpy as np
import time
import math
import OCR_Layers as nn
import OCR_IDX as idx

def randomBatch(data, batchSize):
	indices = np.random.randint(len(data),size=batchSize)
	batch = [data[i] for i in indices]
	return batch


if __name__ == "__main__":
	trainingImageFilename = "../MNIST/train-images.idx3-ubyte"
	trainingLabelFilename = "../MNIST/train-labels.idx1-ubyte"
	testImageFilename = "../MNIST/t10k-images.idx3-ubyte"
	testLabelFilename = "../MNIST/t10k-labels.idx1-ubyte"
	imageSize = [28,28]
	numDigits = 10
	trainingSteps = 3000
	testSteps = -1 #Any negative number for all available test data.
	batchSize = 50
	#stepSize = 0.1
	stepSize = 0.0001
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	testConsoleUpdate = 100
	
	inputSize = imageSize[0]*imageSize[1]
	priorLow = 0.001
	priorHigh = 0.01
	
	
	net = nn.NeuralNet(stepSize, beta1, beta2, epsilon)
	testArgs = []
	
	cnn = "google"
	
	if cnn == "convtest":
		#Convolution layer 1
		inputSize = (1, imageSize[0], imageSize[1])
		nodesOut = 1
		windowSize = (5,5)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/windowSize[0]*windowSize[1]))
		layer1 = nn.Convolutional(inputSize, nodesOut, windowSize, normPriorFun)
		net.appendHLayer(layer1)
		testArgs.append(())
		
		#Maxpool layer 1
		size = (1, imageSize[0], imageSize[1])
		factor = 2
		layer2 = nn.MaxPool(size, factor)
		net.appendHLayer(layer2)
		testArgs.append(())
		
		#Recifying layer 1
		size = int(imageSize[0]/factor)*int(imageSize[1]/factor)
		layer3 = nn.LinRec(size)
		net.appendHLayer(layer3)
		testArgs.append(())
		
		#Linear layer 1
		size = (int(imageSize[0]*imageSize[1]/factor**2), numDigits)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size[0]))
		layer4 = nn.Linear(size, normPriorFun)
		net.appendHLayer(layer4)
		testArgs.append(())
		
		#Output layer
		outputLayer = nn.SoftmaxRelativeEntropy(numDigits)
		net.outputLayer = outputLayer
	
	if cnn == "convtest2":
		
		#Convolution layer 1
		inputSize = (1, 28,28)
		nodesOut = 8
		windowSize = (5,5)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/windowSize[0]*windowSize[1]))
		layer1 = nn.Convolutional(inputSize, nodesOut, windowSize, normPriorFun)
		net.appendHLayer(layer1)
		testArgs.append(())
		
		#Maxpool layer 1
		size = (8, 28, 28)
		factor = 2
		layer2 = nn.MaxPool(size, factor)
		net.appendHLayer(layer2)
		testArgs.append(())
		
		#Recifying layer 1
		size = 8*14*14
		layer3 = nn.LinRec(size)
		net.appendHLayer(layer3)
		testArgs.append(())
		
		#Convolution layer 2
		inputSize = (8, 14, 14)
		nodesOut = 16
		windowSize = (5,5)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/windowSize[0]*windowSize[1]))
		layer4 = nn.Convolutional(inputSize, nodesOut, windowSize, normPriorFun)
		net.appendHLayer(layer4)
		testArgs.append(())
		
		#Maxpool layer 2
		size = (16, 14, 14)
		factor = 2
		layer5 = nn.MaxPool(size, factor)
		net.appendHLayer(layer5)
		testArgs.append(())
		
		#Recifying layer 2
		size = 16*7*7
		layer6 = nn.LinRec(size)
		net.appendHLayer(layer6)
		testArgs.append(())
		
		#Dropout layer 1
		dropoutFractionTrain = 0.5
		dropoutFractionTest = 0.0
		layer9 = nn.Dropout(size, dropoutFractionTrain)
		net.appendHLayer(layer9)
		testArgs.append((dropoutFractionTest,))
		
		#Linear layer 1
		size = (16*7*7, numDigits)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size[0]))
		layer7 = nn.Linear(size, normPriorFun)
		net.appendHLayer(layer7)
		testArgs.append(())
		
		#Output layer
		outputLayer = nn.SoftmaxRelativeEntropy(numDigits)
		net.outputLayer = outputLayer
	
	if cnn == "convtest3":
		#Linear layer 1
		inputSize = imageSize[0]*imageSize[1]
		size = (inputSize, inputSize)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/inputSize))
		layer1 = nn.Linear(size,normPriorFun)
		net.appendHLayer(layer1)
		testArgs.append(())
		
		#Convolution layer 1
		inputSize = (1, imageSize[0], imageSize[1])
		nodesOut = 1
		windowSize = (5,5)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/windowSize[0]*windowSize[1]))
		layer2 = nn.Convolutional(inputSize, nodesOut, windowSize, normPriorFun)
		net.appendHLayer(layer2)
		testArgs.append(())
		
		#Maxpool layer 1
		size = (1, imageSize[0], imageSize[1])
		factor = 2
		layer3 = nn.MaxPool(size, factor)
		net.appendHLayer(layer3)
		testArgs.append(())
		
		#Recifying layer 1
		size = int(imageSize[0]/factor)*int(imageSize[1]/factor)
		layer4 = nn.LinRec(size)
		net.appendHLayer(layer4)
		testArgs.append(())
		
		#Linear layer 1
		size = (int(imageSize[0]*imageSize[1]/factor**2), numDigits)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size[0]))
		layer5 = nn.Linear(size, normPriorFun)
		net.appendHLayer(layer5)
		testArgs.append(())
		
		#Output layer
		outputLayer = nn.SoftmaxRelativeEntropy(numDigits)
		net.outputLayer = outputLayer
	
	if cnn == "linear":
		#Linear layer 1
		size = (imageSize[0]*imageSize[1], numDigits)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size[0]))
		layer = nn.Linear(size, normPriorFun)
		net.appendHLayer(layer)
		
		#Output layer
		outputLayer = nn.SoftmaxRelativeEntropy(numDigits)
		net.outputLayer = outputLayer
		
	if cnn == "linear2":
		#Linear layer 1
		size = (imageSize[0]*imageSize[1], numDigits)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size[0]))
		layer1 = nn.Linear(size, normPriorFun)
		net.appendHLayer(layer1)
		
		#Linear layer 2
		size = (numDigits, numDigits)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size[0]))
		layer2 = nn.Linear(size, normPriorFun)
		net.appendHLayer(layer2)
		
		#Output layer
		outputLayer = nn.SoftmaxRelativeEntropy(numDigits)
		net.outputLayer = outputLayer	
	
	if cnn == "linear3":
		#Linear layer 1
		inputSize = imageSize[0]*imageSize[1]
		size = (inputSize, inputSize)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/inputSize))
		layer1 = nn.Linear(size,normPriorFun)
		net.appendHLayer(layer1)
		testArgs.append(())
		
		#Maxpool layer 1
		size = (1, imageSize[0], imageSize[1])
		factor = 2
		layer2 = nn.MaxPool(size, factor)
		net.appendHLayer(layer2)
		testArgs.append(())
		
		#Recifying layer 1
		size = int(imageSize[0]/factor)*int(imageSize[1]/factor)
		layer3 = nn.LinRec(size)
		net.appendHLayer(layer3)
		testArgs.append(())
		
		#Linear layer 1
		size = (int(imageSize[0]*imageSize[1]/factor**2), numDigits)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size[0]))
		layer4 = nn.Linear(size, normPriorFun)
		net.appendHLayer(layer4)
		testArgs.append(())
		
		#Output layer
		outputLayer = nn.SoftmaxRelativeEntropy(numDigits)
		net.outputLayer = outputLayer
	
	if cnn == "google":
		#Convolution layer 1
		inputSize = (1, imageSize[0], imageSize[1])
		nodesOut = 32
		windowSize = (5,5)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/windowSize[0]*windowSize[1]))
		layer1 = nn.Convolutional(inputSize, nodesOut, windowSize, normPriorFun)
		net.appendHLayer(layer1)
		testArgs.append(())
	
		#Maxpool layer 1
		size = (32, imageSize[0], imageSize[1])
		factor = 2
		layer2 = nn.MaxPool(size, factor)
		net.appendHLayer(layer2)
		testArgs.append(())
		
		#Recifying layer 1
		size = 32*int(imageSize[0]/factor)*int(imageSize[1]/factor)
		layer3 = nn.LinRec(size)
		net.appendHLayer(layer3)
		testArgs.append(())
	
	
		#Convolution layer 2
		inputSize = (32, int(imageSize[0]/factor), int(imageSize[1]/factor))
		nodesOut = 64
		windowSize = (5,5)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/windowSize[0]*windowSize[1]))
		layer4 = nn.Convolutional(inputSize, nodesOut, windowSize, normPriorFun)
		net.appendHLayer(layer4)
		testArgs.append(())
	
		#Maxpool layer 2
		size = (64,  int(imageSize[0]/factor), int(imageSize[1]/factor))
		factor = 2
		layer5 = nn.MaxPool(size, factor)
		net.appendHLayer(layer5)
		testArgs.append(())
	
	
		#Recifying layer 2
		size = 64*int(imageSize[0]/factor**2)*int(imageSize[1]/factor**2)
		layer6 = nn.LinRec(size)
		net.appendHLayer(layer6)
		testArgs.append(())
	
		#Linear layer 1
		linearOutputSize = 1024
		linearSize = (size, linearOutputSize)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size))
		layer7 = nn.Linear(linearSize, normPriorFun)
		net.appendHLayer(layer7)
		testArgs.append(())
	
		#Recifying layer 3
		size = 1024
		layer8 = nn.LinRec(size)
		net.appendHLayer(layer8)
		testArgs.append(())
		
		#Dropout layer 1
		dropoutFractionTrain = 0.5
		dropoutFractionTest = 0.0
		layer9 = nn.Dropout(size, dropoutFractionTrain)
		net.appendHLayer(layer9)
		testArgs.append((dropoutFractionTest,))
	
		#Linear layer 2
		linearSize = (size, numDigits)
		normPriorFun = (lambda : np.random.uniform(priorLow, priorHigh)*np.sqrt(2.0/size))
		layer10 = nn.Linear(linearSize, normPriorFun)
		net.appendHLayer(layer10)
		testArgs.append(())
	
		#Output layer
		outputLayer = nn.SoftmaxRelativeEntropy(numDigits)
		net.outputLayer = outputLayer
	
	timeStart = time.time()
	
	print("Loading training data from file.")
	trainingData = idx.getPreparedMNISTData(trainingImageFilename, trainingLabelFilename)
	print("Loading test data from file.")
	testData = idx.getPreparedMNISTData(testImageFilename, testLabelFilename)
	
	timeLoad = time.time()
	

	
	
	
	print("Training.")
	
	counter = 0
	
	for i in range(trainingSteps):
		counter = counter + 1
		startTime = time.time()
		print("On training step: " + str(counter))
		batch = randomBatch(trainingData, batchSize)
		loss, norm = net.train(batch)
		currentTime = time.time()
		#print("Time taken: " + "{:.2f}".format(currentTime - startTime) + " Batch loss: " + "{:.2f}".format(loss) + " Norm: " + "{:.2g}".format(norm))
		print("Time taken: " + "{:.2f}".format(currentTime - startTime) + " Batch loss: " + "{:.2f}".format(loss))
			
			
	net.printTimes()	
		
	timeTrain = time.time()
	
	print("Testing.")
	
	net.setArgs(testArgs)
	
	numPassed = 0
	counter = 0
	startTime = time.time()
	for test in testData:
		counter = counter + 1
		
		
		prediction = net.infer(test[0])
		actual = np.argmax(test[1])
		if prediction == actual:
			numPassed = numPassed + 1
			
		if counter % testConsoleUpdate == 0:
			currentTime = time.time()
			print("Finished test " + str(counter) + ".  Time since last update: " + "{:.2f}".format(currentTime - startTime))	
			startTime = time.time()
			
		if testSteps >= 0 and counter >= testSteps:
			break
			
	if testSteps >= 0:
		numTested = testSteps
	else:
		numTested = len(testData)
		
	fractionPassed = numPassed / numTested
	
	print("Fraction passed: " + str(fractionPassed))
