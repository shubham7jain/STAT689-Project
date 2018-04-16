import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import random
from struct import unpack
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

## computes the p_i for any x_i and beta
def logistic(x,beta):
    return( 1 / (1+np.exp(-np.sum(x*beta))))

def generate_mislabeled_data(n, alpha):
    x = np.linspace(-5,5,num=n)
    beta = np.array([1.,1.])
    X = np.column_stack((np.ones(n),x))

    p = np.apply_along_axis(lambda y: logistic(y,beta),1,X)

    ## simulate responses
    y = np.random.binomial(n=1,p=p)

    ## contamination level alpha
    z = np.random.binomial(n=1,p=1-alpha,size=n)

    r = np.random.binomial(n=1,p=0.5,size=n)

    ## we observe mnist-data w
    w = y*z + r*(1-z)

    return X, w

def loadmnist(imagefile, labelfile):

    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get mnist-data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)

def read_mnist_data(alpha, corrupted):
    train_img, train_lbl = loadmnist('mnist-data/train-images-idx3-ubyte'
                                     , 'mnist-data/train-labels-idx1-ubyte')
    test_img, test_lbl = loadmnist('mnist-data/t10k-images-idx3-ubyte'
                                   , 'mnist-data/t10k-labels-idx1-ubyte')

    train_img = train_img[:10000]
    train_lbl = train_lbl[:10000]
    test_img = test_img[:1000]
    test_lbl = test_lbl[:1000]

    for i in range(len(train_lbl)):
        if (random.uniform(0, 10) <= alpha*10):
            train_lbl[i] = random.randint(1, 10)
            corrupted.append(i)

    return train_img, test_img, train_lbl, test_lbl

def read_iris_data(alpha):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

    for i in range(len(y_train)):
        if (random.uniform(0, 3) <= alpha*3):
            y_train[i] = random.randint(0, 2)

    return X_train, X_test, y_train, y_test
