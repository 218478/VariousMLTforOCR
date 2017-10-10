# WCZYTYWANIE NIE JEST MOJE WIEC ZROB SWOJE ALBO TEN KOD PO PROSTU PRZEROB


import os, logging, time, struct, argparse
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def readMNIST(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def printInfoToConsole(image):
    """
    Info about number and image dimensions.
    """
    print image[0]
    print "Dim: " + str(len(image[1])) + " x " + str(len(image[1][1]))

def getHorizontalAndVerticalHistogram(image):
    """
    Returns two vectors. First is a horizontal histogram and
    the second one is a vertical histogram.
    """
    vertical = [0]*len(image[1]) # initialized with zeros
    horizontal = [0]*len(image[1][0]) # initialized with zeros
    for i in xrange(0, len(image[1])):
        for j in xrange(0, len(image[1][i])):
            horizontal[j] += image[1][i][j]
            vertical[i] += image[1][i][j]

    print vertical
    print horizontal
    return vertical, horizontal

def binarize(image):
    """
    Binarizes the MNIST image. Usage: binarize(image)
    """
    for i in xrange(0, len(image[1])):
        for j in xrange(0, len(image[1][i])):
            if image[1][i][j] > 0:
                image[1][i][j] = 1


def displayNImages(n, pathToDatasets):
    """
    Displays info about N images.
    """
    images = readMNIST(path=pathToDatasets)
    i = 0
    kNNforMNIST(images, n = 1) # achieved 67.7 % 
    for image in images:
        # show(image[1])
        # printInfoToConsole(image)
        # binarize(image)
        # getHorizontalAndVerticalHistogram(image)
        i += 1
        # if i > n-1:
        #     break

    print i

def countSimilarity(image, patternSet, binarized = True):
    """
    Returns sorted array of closest characters.
    """
    patternArray = []

    if binarized:
        for charNo, characterSet in enumerate(patternSet):
            for patternImage in characterSet:
                temp = patternImage.reshape(28, 28) * image
                patternArray.append([charNo, sum(sum(temp))])
    else:
        for charNo, characterSet in enumerate(patternSet):
            for patternImage in characterSet:
                temp = np.linalg.norm(patternImage.reshape(28, 28) - image)
                patternArray.append([charNo, temp])


    # sorted(patternArray, key = lambda patternElem: patternElem[1] )
    patternArray.sort(key= lambda patternElem: patternElem[1], reverse=True )
    return patternArray

def chooseNeighbor(simArray, n = 7):
    a = [0]*10 # change to np.zeros()
    for i in xrange(0, n):
        a[simArray[i][0]] += 1
    return a.index(max(a))

def kNNforMNIST(images, n):
    mySet = createPatternSetForKNN(images, 10, n = 10)
    successNo = 0
    failuresNo = 0

    start = time.clock()
    for img in images:
        img = (next(images))
        # binarize(img)
        sim = countSimilarity(img[1], mySet, binarized = True)
        neighbor = chooseNeighbor(sim, n)
        if neighbor == img[0]:
            successNo += 1
        else:
            failuresNo += 1

        if (failuresNo + successNo) > 1000:
            break

    stop = time.clock()
    print "skutecznosc: " + str(float(successNo)/(successNo+failuresNo))
    print "czas calkowity: " + str(stop-start)
    print "czas na probke: " + str(float(stop-start)/(successNo+failuresNo))
    exit()

# TODO: change the name of n variable
def createPatternSetForKNN(images, classesNo, n=10):
    """
    Creates pattern for kNN image recognition. Parameter n tells
    how many role models should the algorithm base for each class.
    """

    howManyLeft = [n]*classesNo
    imageDimension = len(next(images)[1])*len(next(images)[1][0])
    patternSet = np.ones((classesNo, n, imageDimension), dtype=int) # creates a set of classesNo tuples
    while sum(howManyLeft) > 0:
        image = next(images)
        binarize(image)
        logging.info("image binarized")
        if howManyLeft[image[0]] > 0:
            patternSet[image[0]][n-howManyLeft[image[0]]] = np.ravel(image[1])
            howManyLeft[image[0]] -= 1

    return patternSet
    
if __name__ == '__main__':
    # from appJar import gui
    # # create a GUI variable called app
    # app = gui()
    # app.addLabel("title", "Welcome to appJar")
    # app.setLabelBg("title", "red")
    # app.go()

    parser = argparse.ArgumentParser()
    parser.add_argument("pathToDatasets", help="Directory to stored datasets")
    parser.add_argument("pathToLogFile", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFile, 'logFile.log'))#, level = logging.INFO)

    displayNImages(1, args.pathToDatasets)