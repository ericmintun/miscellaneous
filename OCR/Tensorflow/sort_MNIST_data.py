import OCR_MNIST as mnist
import numpy as np

def main():
    directory = "../../MNIST/"
    trainImagesName = directory + 'train-images.idx3-ubyte.gz'
    trainLabelsName = directory + 'train-labels.idx1-ubyte.gz'
    testImagesName = directory + 't10k-images.idx3-ubyte.gz'
    testLabelsName = directory + 't10k-labels.idx1-ubyte.gz'

    sortedTrainImagesName = directory + 'train-images-sorted.idx3-ubyte.gz'
    sortedTrainLabelsName = directory + 'train-labels-sorted.idx1-ubyte.gz'
    sortedTestImagesName = directory + 't10k-images-sorted.idx3-ubyte.gz'
    sortedTestLabelsName = directory + 't10k-labels-sorted.idx1-ubyte.gz'

    with open(trainImagesName, 'rb') as f:
        trainImages = mnist.extract_images(f)

    with open(trainLabelsName, 'rb') as f:
        trainLabels = mnist.extract_labels(f)

    trainSortedIndices = np.argsort(trainLabels, kind='mergesort')

    sortedTrainLabels = trainLabels[trainSortedIndices]
    sortedTrainImages = trainImages[trainSortedIndices]

    with open(sortedTrainImagesName, 'wb') as f:
        mnist.write_images(f, sortedTrainImages)

    with open(sortedTrainLabelsName, 'wb') as f:
        mnist.write_labels(f, sortedTrainLabels)


    with open(testImagesName, 'rb') as f:
        testImages = mnist.extract_images(f)

    with open(testLabelsName, 'rb') as f:
        testLabels = mnist.extract_labels(f)

    testSortedIndices = np.argsort(testLabels, kind='mergesort')

    sortedTestLabels = testLabels[testSortedIndices]
    sortedTestImages = testImages[testSortedIndices]

    with open(sortedTestImagesName, 'wb') as f:
        mnist.write_images(f, sortedTestImages)

    with open(sortedTestLabelsName, 'wb') as f:
        mnist.write_labels(f, sortedTestLabels)

    """
    with open(sortedTrainImagesName, 'rb') as f:
        testSortedImages = mnist.extract_images(f).reshape((60000,28,28))

    with open(sortedTrainLabelsName, 'rb') as f:
        testSortedLabels = mnist.extract_labels(f)

    print(testSortedImages[11999])
    print(testSortedLabels[::3000])
    unique, counts = np.unique(testSortedLabels, return_counts = True)
    numEach = dict(zip(unique, counts))
    print(numEach)
    """

if __name__ == "__main__":
    main()

