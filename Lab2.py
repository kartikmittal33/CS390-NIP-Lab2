import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ALGORITHM = "guesser"
# ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

DATASET = "mnist_d"
# DATASET = "mnist_f"
# DATASET = "cifar_10"
# DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3

NEURONS = 1024


# =========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.05):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # TODO: implement

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        sig = self.__sigmoid(x)
        der = sig * (1 - sig)
        return der  # TODO: implement

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    # TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
    def train(self, xVals, yVals, epochs=5, minibatches=True, mbs=100):
        num_batches = xVals.shape[0] / mbs
        xValBatches = np.split(xVals, num_batches)
        yValBatches = np.split(yVals, num_batches)
        for i in range(epochs):
            for j in range(num_batches):
                layer1, layer2 = self.__forward(xValBatches[j])
                L2e = (layer2 - yValBatches[j])

                sig_der_layer2 = self.__sigmoidDerivative(layer2)
                L2d = L2e * sig_der_layer2

                L1e = np.dot(L2d, self.W2.T)

                sig_der_layer1 = self.__sigmoidDerivative(layer1)
                L1d = L1e * sig_der_layer1

                L1a = (np.dot(xValBatches[j].T, L1d)) * self.lr
                L2a = (np.dot(layer1.T, L2d)) * self.lr

                self.W1 -= L1a
                self.W2 -= L2a

        return self

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        ans = []
        for entry in layer2:
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            index = entry.argmax()
            pred[index] = 1
            ans.append(pred)

        return np.array(ans)


def buildTFNeuralNet(x, y, eps=6):
    # TODO: Implement a standard ANN here.

    model = NeuralNetwork_2Layer(IS, NUM_CLASSES, NEURONS)

    return model.train(x, y)


def buildTFConvNet(x, y, eps=10, dropout=True, dropRate=0.2):
    # TODO: Implement a CNN here. dropout option is required.
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(IH, IW, IZ)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
         tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
         tf.keras.layers.Dropout(dropRate),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(NEURONS, activation=tf.nn.relu),
         tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=eps)
    return model


# =========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist_f = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist_f.load_data()
    elif DATASET == "cifar_10":
        cifar10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    elif DATASET == "cifar_100_f":
        cifar100_f = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100_f.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        cifar100_c = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100_c.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)

    # xTrain = xTrain[:1000]
    # yTrain = yTrain[:1000]

    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / (preds.shape[0] * 1.0)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
