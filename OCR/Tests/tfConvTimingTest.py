import tensorflow as tf
import time

def main():
    i = tf.random_uniform([100,28,28,10])
    w = tf.random_uniform([5,5,10,10])

    c = tf.nn.conv2d(i,w,[1,1,1,1],"SAME")

    sess = tf.Session()
    #h = sess.partial_run_setup([i,w,c])

    #genTime = 0.0
    #convTime = 0.0

    startTime = time.time()
    for i in range(1):
        sess.run(c)
        #currentTime = time.time()
        #sess.partial_run(h,[i,w])
        #genTime = genTime + time.time() - currentTime

        #currentTime = time.time()
        #sess.partial_run(h,c)
        #convTime = convTime + time.time() - currentTime
    totalTime = time.time() - startTime
    #totalTime = genTime + convTime

    print("Total time: " + str(totalTime) + "\n")
    #print("Convolve time: " + str(convTime) + "\n")
    #print("Generation time: " + str(genTime) + "\n")

if __name__ == "__main__":
    main()
