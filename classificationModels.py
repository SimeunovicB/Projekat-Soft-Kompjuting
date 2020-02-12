from sklearn.neighbors import KNeighborsClassifier  # KNN
import pickle
import numpy
import tensorflow as tf
from joblib import load, dump
import cv2 as cv
from keras import optimizers
from keras.layers import LeakyReLU
import glob
from keras.models import Sequential
from keras.layers.core import Dense, Activation


new_dataset_location = './datasets/cifar-10-batches-py'


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def loadGray(path):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.resize(image, (64, 64))


def reshape_data(input_data):
    input_data = numpy.array(input_data)
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def getData():
    """labels = []
    images = []
    for i in range(1, 6):
        d = unpickle(new_dataset_location+'/data_batch_'+str(i))
        for id in range(0, len(d[b'labels'])):
            if d[b'labels'][id] == 0 or d[b'labels'][id] == 9:
                labels.append(d[b'labels'][id])
                images.append(d[b'data'][id])

    images = numpy.array(images).reshape((len(images), 32, 32, 3))
    imgs = []
    for im in images:
        imgs.append(cv.resize(cv.cvtColor(im, cv.COLOR_RGB2GRAY),
                              (64, 64), cv.INTER_CUBIC))

    HOG = cv.HOGDescriptor((64, 64), (32, 32), (8, 8), (8, 8), 9)

    images = []
    for im in imgs:
        images.append(HOG.compute(im))
    """
    images = []
    labels = []
    HOG = cv.HOGDescriptor((64, 64), (32, 32), (8, 8), (8, 8), 9)
    for image_path in glob.glob('./datasets/car/' + "*.jpg"):
        img = loadGray(image_path)
        computedHOG = HOG.compute(img)
        images.append(computedHOG)
        labels.append(0)
    for image_path in glob.glob('./datasets/truck/' + "*.jpg"):
        img = loadGray(image_path)
        computedHOG = HOG.compute(img)
        images.append(computedHOG)
        labels.append(1)

    return images, labels


def getKNN():
    try:
        return load('./knn.knn')
    except:
        pass
    images, labels = getData()

    clf_knn = KNeighborsClassifier(n_neighbors=50)
    clf_knn = clf_knn.fit(reshape_data(images), numpy.array(labels))
    try:
        dump(clf_knn, './knn.knn')
    except:
        pass
    return clf_knn


def getNetwork():
    network = Sequential()
    network.add(Dense(1200, input_dim=3600))
    network.add(LeakyReLU(alpha=0.1))
    network.add(Dense(384))

    network.add(LeakyReLU(alpha=0.01))
    network.add(Dense(2, activation='softmax'))
    # network.summary()
    try:
        network.load_weights('./network.h5')
    except:
        images, labels = getData()
        network.compile(
            optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
        network.fit(reshape_data(images), numpy.array(labels),
                    batch_size=30, epochs=70, verbose=1)
        network.save_weights('./network.h5')
    return network
