import OCR_MNIST as mnist
import tensorflow as tf
import numpy as np
import time

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def main():
    directory = "../../MNIST/"
    trainImagesFilename = directory + 'train-images.idx3-ubyte.gz'
    trainLabelsFilename = directory + 'train-labels.idx1-ubyte.gz'
    testImagesFilename = directory + 't10k-images.idx3-ubyte.gz'
    testLabelsFilename = directory + 't10k-labels.idx1-ubyte.gz'

    trainData = mnist.MNISTPairs(trainImagesFilename, trainLabelsFilename, 0.5)
    testData = mnist.MNISTPairs(testImagesFilename, testLabelsFilename, 0.5)


    x1 = tf.placeholder(tf.float32, shape=[None, 28*28])
    x2 = tf.placeholder(tf.float32, shape=[None, 28*28])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    keep_prob = tf.placeholder(tf.float32)

    W_conv1 = weight_variable([5,5,1,16])
    b_conv1 = bias_variable([16])

    x1_image = tf.reshape(x1, [-1,28,28,1])
    x2_image = tf.reshape(x2, [-1,28,28,1])

    h_conv11 = tf.nn.relu(conv2d(x1_image, W_conv1) + b_conv1)
    h_pool11 = max_pool_2x2(h_conv11)

    h_conv12 = tf.nn.relu(conv2d(x2_image, W_conv1) + b_conv1)
    h_pool12 = max_pool_2x2(h_conv12)

    W_c11 = weight_variable([14*14*16,1024])
    W_c12 = weight_variable([14*14*16,1024])
    b_c1 = bias_variable([1024])

    h_pool11_flat = tf.reshape(h_pool11, [-1,14*14*16])
    h_pool12_flat = tf.reshape(h_pool12, [-1,14*14*16])
    h_c1 = tf.nn.relu(tf.matmul(h_pool11_flat, W_c11) + tf.matmul(h_pool12_flat, W_c12) + b_c1)

    h_c1_drop = tf.nn.dropout(h_c1, keep_prob)


    W_conv2 = weight_variable([5,5,16,32])
    b_conv2 = bias_variable([32])


    h_conv21 = tf.nn.relu(conv2d(h_pool11, W_conv2) + b_conv2)
    h_pool21 = max_pool_2x2(h_conv21)

    h_conv22 = tf.nn.relu(conv2d(h_pool12, W_conv2) + b_conv2)
    h_pool22 = max_pool_2x2(h_conv22)

    
    W_c21 = weight_variable([7*7*32,1024])
    W_c22 = weight_variable([7*7*32,1024])
    b_c2 = bias_variable([1024])

    h_pool21_flat = tf.reshape(h_pool21, [-1,7*7*32])
    h_pool22_flat = tf.reshape(h_pool22, [-1,7*7*32])
    h_c2 = tf.nn.relu(tf.matmul(h_pool21_flat, W_c21) + tf.matmul(h_pool22_flat, W_c22) + b_c2)

    h_c2_drop = tf.nn.dropout(h_c2, keep_prob)
    
    W_fc1 = weight_variable([1024,2])
    W_fc2 = weight_variable([1024,2])
    b_c2 = bias_variable([2])

    y_conv = tf.matmul(h_c1_drop, W_fc1) + tf.matmul(h_c2_drop, W_fc2) + b_c2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))


    excludeOnTrain = [3,6]
    #stepSize = tf.placeholder(tf.float32)

    init = tf.global_variables_initializer()

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #train_step = tf.train.GradientDescentOptimizer(stepSize).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y_,1),tf.argmax(y_conv,1),num_classes=2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        currentTime = time.time()
        #batchingTime = 0
        print("Training without the followings digits:")
        print(excludeOnTrain)
        for i in range(200000):
            #batchingTimeS = time.time()
            batch = trainData.nextBatch(50, excluded=excludeOnTrain)
            #batchingTime += time.time() - batchingTimeS
            if i%100 == 0:
                timeTaken = time.time() - currentTime
                currentTime = time.time()
                train_accuracy = accuracy.eval(feed_dict={x1:batch[0], x2:batch[1], y_: batch[2], keep_prob: 1.0})
                print("Step %d, training accuracy %g, time taken %g"  %(i, train_accuracy, timeTaken))
                #batchingTime = 0
            train_step.run(feed_dict={x1: batch[0], x2:batch[1], y_: batch[2], keep_prob: 0.5})

        

        rollingAccuracy = 0.0
        rolling_confusion_matrix = np.zeros((2,2))
        testNum = 10000
        print("Testing without the following digits:")
        print(excludeOnTrain)
        for i in range(testNum):
            testBatch = testData.nextBatch(50, excluded=excludeOnTrain)
            #test_accuracy = accuracy.eval(feed_dict={x1: testBatch[0], x2:testBatch[1],  y_:testBatch[2], keep_prob: 1.0})
            test_confusion_matrix = confusion_matrix.eval(feed_dict={x1: testBatch[0], x2:testBatch[1],  y_:testBatch[2], keep_prob: 1.0})
            #rollingAccuracy += test_accuracy
            rolling_confusion_matrix += test_confusion_matrix
            if i%100 == 0:
                print("Test step %d" % i)
                #print(test_confusion_matrix)

        confusion_norms = np.sum(rolling_confusion_matrix, axis=1)
        confusion_matrix_normed = (rolling_confusion_matrix.T/confusion_norms).T
        test_accuracy = np.trace(rolling_confusion_matrix)/np.sum(rolling_confusion_matrix)

        #rollingAccuracy = rollingAccuracy / testNum
        print("Overall test accuracy %g" % (test_accuracy))
        print("Overall confusion matrix.  Has the form:")
        print("[[Equal and predicts equal, equal but predicts different],")
        print("[Different but predicts equal, different and predicts different]]")
        print(confusion_matrix_normed)


        rollingAccuracy = 0.0
        rolling_confusion_matrix = np.zeros((2,2))
        testNum = 10000
        print("Testing requiring at both of the following digits:")
        print(excludeOnTrain)
        for i in range(testNum):
            testBatch = testData.nextBatch(50, required=excludeOnTrain,bothRequired=True)
            #test_accuracy = accuracy.eval(feed_dict={x1: testBatch[0], x2:testBatch[1],  y_:testBatch[2], keep_prob: 1.0})
            test_confusion_matrix = confusion_matrix.eval(feed_dict={x1: testBatch[0], x2:testBatch[1],  y_:testBatch[2], keep_prob: 1.0})
            #rollingAccuracy += test_accuracy
            rolling_confusion_matrix += test_confusion_matrix
            if i%100 == 0:
                print("Test step %d" % i)
                #print(test_confusion_matrix)

        confusion_norms = np.sum(rolling_confusion_matrix, axis=1)
        confusion_matrix_normed = (rolling_confusion_matrix.T/confusion_norms).T
        test_accuracy = np.trace(rolling_confusion_matrix)/np.sum(rolling_confusion_matrix)

        #rollingAccuracy = rollingAccuracy / testNum
        print("Overall test accuracy %g" % (test_accuracy))
        print("Overall confusion matrix.  Has the form:")
        print("[[Equal and predicts equal, equal but predicts different],")
        print("[Different but predicts equal, different and predicts different]]")
        print(confusion_matrix_normed)


if __name__ == "__main__":
    main()
