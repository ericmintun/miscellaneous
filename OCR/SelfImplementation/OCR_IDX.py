import struct
import numpy as np
#import png

def splicer(data, spliceSize):
	#counter = 0
	for i in range(0,len(data),spliceSize):
		#counter = counter + 1
		#if counter % 1000000 == 0:
		#	print("On byte " + str(counter)) 
		yield data[i:i+spliceSize]

def readHeader(header):
	magicNumbers = list(header)
	dataType = magicNumbers[2]
	numDim = magicNumbers[3]
	if dataType == 8:
		byteSize = 1
		unpackParam = '>B'
	elif dataType == 9:
		byteSize = 1
		unpackParam = '>b'
	elif dataType == 11:
		byteSize = 2
		unpackParam = '>h'
	elif dataType == 12:
		byteSize = 4
		unpackParam = '>i'
	elif dataType == 13:
		byteSize = 4
		unpackParam = '>f'
	elif dataType == 14:
		byteSize = 8
		unpackParam = '>d'
	else:
		raise IOError('Invalid data type specified in IDX file.')
		
	return numDim, byteSize, unpackParam		
		
def readIDXFile(filename):
	with open(filename, 'rb') as f:
		magicNumber = f.read(4)
		numDim, byteSize, unpackParam = readHeader(magicNumber)
			
		arrayDims = []
		for i in range(numDim):
			arrayDims.append(struct.unpack('>i', f.read(4))[0])
			
		#print(byteSize)
		#print(unpackParam)
		#print(arrayDims)
			
		#totalData = reduce(mul, arrayDims)
		data = f.read()
		dataArray = np.array(list(map(lambda x : struct.unpack(unpackParam, x)[0], splicer(data,byteSize))))
		return np.reshape(dataArray, tuple(arrayDims))	
		
def readMNISTData(imageFilename, labelFilename):
	imageData = readIDXFile(imageFilename)
	labelData = readIDXFile(labelFilename)
	if len(imageData) != len(labelData):
		raise ValueError('The number of labels and number of images do not match.')
	return list(zip(imageData,labelData))
	
def getPreparedMNISTData(imageFilename, labelFilename):
	imageData = readIDXFile(imageFilename)/255
	labelDataRaw = readIDXFile(labelFilename)
	labelData = np.zeros((len(labelDataRaw),10))
	for i in range(len(labelDataRaw)):
		labelData[i,labelDataRaw[i]]=1
		
	if len(imageData) != len(labelData):
		raise ValueError('The number of labels and number of images do not match.')
	return list(zip(imageData,labelData))

#def writeIDXFile(filename, data):
	
	
#Returns a generator of the highest level objects in the IDX file.
#def genIDXElement(filename):
#	with open(filename, 'rb') as f:
#		magicNumber = f.read(4)
#		numDim, byteSize, unpackParam = readHeader(magicNumber)
#		
#		arrayDims = []
#		for i in range(numDim):
#			arrayDims.append(struct.unpack('>i', f.read(4))[0])
#		
#		dataSize = byteSize
#		for dim in arrayDims[1:]:
#			dataSize = dataSize * dim
#		
#		for i in range(arrayDims[0]):
#			currentData = f.read(dataSize)
#			dataArray = np.array(list(map(lambda x : struct.unpack(unpackParam, x)[0], splicer(currentData,byteSize))))
#			yield np.reshape(dataArray, tuple(arrayDims[1:]))	
	
#def genMNISTData(imageFilename, labelFilename):
#	for label, image in zip(genIDXElement(labelFilename),genIDXElement(imageFilename)):
#		yield (image, label)

#if __name__ == "__main__":
#	imageFilename = "..\\MNIST\\t10k-images.idx3-ubyte"
#	labelFilename = "..\\MNIST\\t10k-labels.idx1-ubyte"
#	
#	c=0
#	for j,i in genMNISTData(imageFilename, labelFilename):
#		c = c+1
#		print(i)
#		f = open('testMNISTpng' + str(c) + '.png', 'wb')
#		w = png.Writer(28,28,greyscale=True,bitdepth=8)
#		w.write(f,j.tolist())
#		f.close()
#		if c > 3:
#			break