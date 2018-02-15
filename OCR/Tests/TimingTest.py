import numpy as np
import time

if __name__ == "__main__":
	test = np.random.rand(100000)
	time1 = time.time()
	test**2
	time2 = time.time()
	for x in test:
		x**2
	time3 = time.time()
	print(time2-time1)
	print(time3-time2)