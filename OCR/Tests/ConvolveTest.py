import numpy as np
from scipy.signal import convolve2d
import time

if __name__ == "__main__":
	wSize = (5,5)
	iSize = (28,28)
	num = 1000000
	totalStart = time.time()
	randTime = 0
	convTime = 0
	for i in range(num):
		randStart = time.time()
		window = np.random.random(wSize)
		image = np.random.random(iSize)
		randTime = randTime + time.time() - randStart
		
		convStart = time.time()
		output = convolve2d(image, window, mode='same')
		convTime = convTime + time.time() - convStart
		
	totalTime = time.time() - totalStart
	print("Total time: " + str(totalTime))
	print("Convolution time: " + str(convTime))
	print("Generation time: " + str(randTime))