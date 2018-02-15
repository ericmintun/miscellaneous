#Most of this is stolen directly from tensorflow's implementation.

import numpy as np
import gzip

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def _write32(bytestream, number):
    data = number.to_bytes(4, byteorder='big')
    bytestream.write(data)
    return

def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
        Args:
            f: A file object that can be passed into a gzip reader.
        Returns:
            data: A 4D uint8 numpy array [index, y, x, depth].
        Raises:
            ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.
    Returns:
        labels: a 1D uint8 numpy array.
    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels


def write_images(f, images):
    
    print('Writing', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        num_images = images.shape[0]
        rows = images.shape[1]
        cols = images.shape[2]

        _write32(bytestream, 2051)
        _write32(bytestream, num_images)
        _write32(bytestream, rows)
        _write32(bytestream, cols)

        bytestream.write(memoryview(images))

    return



def write_labels(f, labels):
    
    print('Writing', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        num_labels = labels.shape[0]


        _write32(bytestream, 2049)
        _write32(bytestream, num_labels)

        bytestream.write(memoryview(labels))

    return

def _getRandomPair(labels, excluded=[], required=[],equal=False,replace=False,bothRequired=False):
    

        assert set(excluded).isdisjoint(required)

        allowedLabels = list(labels)
        for exclusion in excluded:
            allowedLabels.remove(exclusion)

        if required != []:
            firstList = required
        else:
            firstList = allowedLabels

        firstValue = np.random.choice(firstList)

        if equal == True:
            secondValue = firstValue
        else:
            if required != [] and bothRequired==True:
                secondList = list(required)
            else:
                secondList = list(allowedLabels)

            if replace == False and firstValue in secondList:
                secondList.remove(firstValue)

            assert secondList
            secondValue = np.random.choice(secondList)

        if np.random.randint(2) == 1:
            return (secondValue, firstValue)
        else:
            return (firstValue, secondValue)




class MNISTPairs:
    
    def __init__(self, imagesFilename, labelsFilename, equalFraction):
        with open(imagesFilename, 'rb') as f:
            imagesUnsorted = extract_images(f)

        with open(labelsFilename, 'rb') as f:
            labelsUnsorted = extract_labels(f)

        assert imagesUnsorted.shape[0] == labelsUnsorted.shape[0]

        sortIndices = np.argsort(labelsUnsorted, kind='mergesort')
        imagesUncut = imagesUnsorted[sortIndices]
        labels = labelsUnsorted[sortIndices]
        
        self.imageSize = imagesUncut.shape[1]*imagesUncut.shape[2]

        assert imagesUncut.shape[3] == 1
        imagesUncut = imagesUncut.reshape((imagesUncut.shape[0],self.imageSize))
        imagesUncut = imagesUncut.astype(np.float32)
        imagesUncut = imagesUncut / 255.0

        uniques, counts = np.unique(labels, return_counts=True)

        self.numChars = len(uniques)
        self.images = {}

        counter = 0
        for unique, count in zip(uniques, counts):
            self.images[unique] = imagesUncut[counter:counter+count]
            counter += count

        self.equalFraction = equalFraction
        self.labels = uniques

    def nextBatch(self, batchSize, excluded=[], required=[], bothRequired=False):


        firstImages = np.zeros((batchSize,self.imageSize))
        secondImages = np.zeros((batchSize, self.imageSize))
        labelsVec = np.zeros((batchSize, 2))


        isEqual = 1 * (np.random.uniform(0.0,1.0,batchSize) < self.equalFraction)


        for i in range(batchSize):
            if(isEqual[i] == 1):
                num = np.random.randint(self.numChars)
                pair = _getRandomPair(self.labels, excluded, required, equal=True)
                labelsVec[i][0] = 1
            else:
                pair = _getRandomPair(self.labels, excluded, required, equal=False, bothRequired=bothRequired)
                labelsVec[i][1] = 1

            firstImages[i] = self.images[pair[0]][np.random.randint(self.images[pair[0]].shape[0])]
            secondImages[i] = self.images[pair[1]][np.random.randint(self.images[pair[1]].shape[0])]
            #secondImages[i] = np.random.choice(self.images[pair[1]],1)

        return firstImages, secondImages, labelsVec


def readWriteTest(directory):
    imagesFilename = directory + 'train-images.idx3-ubyte.gz'
    labelsFilename = directory + 'train-labels.idx1-ubyte.gz'

    imagesOutName = directory + 'images-test.gz'
    labelsOutName = directory + 'labels-test.gz'


    fIn = open(imagesFilename, 'rb')
    images = extract_images(fIn)
    fIn.close()

    fIn = open(labelsFilename, 'rb')
    labels = extract_labels(fIn)
    fIn.close()

    print(labels[0:10])

    fOut = open(imagesOutName, 'wb')
    write_images(fOut, images)
    fOut.close()

    fOut = open(labelsOutName, 'wb')
    write_labels(fOut, labels)
    fOut.close()

    fBackIn = open(imagesOutName, 'rb')
    imagesAgain = extract_images(fBackIn)
    fBackIn.close()

    fBackIn = open(labelsOutName, 'rb')
    labelsAgain = extract_labels(fBackIn)
    fBackIn.close()

    print(labelsAgain[0:10])
    #print(images[0])

    print(np.array_equal(labels, labelsAgain))
    print(np.array_equal(images, imagesAgain))
    print(images.shape)


    

if __name__ == "__main__":
    readWriteTest("../../MNIST/")





