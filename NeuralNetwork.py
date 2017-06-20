from Methods import *
import numpy as np


class NeuralNetwork(Predictor):
    mat = 0
    type = []
    input_labels = []
    inputlayer = 2
    outlayer = 2
    alpha = 0.1
    regularization = 0.01
    relate = {}
    neuralnet = 0
    samples = 0
    epoch=60000


    def __init__(self):
        self.samples = 0  # training set size
        self.inputlayer = 2  # input layer dimensionality
        self.outlayer = 2  # output layer dimensionality
        self.alpha = 0.002  # learning rate for gradient descent
        self.regularization = 0.05  # regularization strength
        self.type = []
        self.relate = {}
        self.mat = 0
        self.input_labels = []
        self.neuralnet = 0
        self.input_label_categories = []

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def result(self, x):
        W1, b1, W2, b2 = self.neuralnet['W1'], self.neuralnet['b1'], self.neuralnet['W2'], self.neuralnet['b2']
        val = np.argmax((np.exp((self.sigmoid(x.dot(W1) + b1)).dot(W2) + b2)) / np.sum((np.exp((self.sigmoid(x.dot(W1) + b1)).dot(W2) + b2)),
                                                                                       axis=1, keepdims=True),axis=1)
        return val
    
    def train(self, instances):
        temp_toto = []
        for instance in instances:
            if instance._label.label_str not in self.type:
                self.type.append(instance._label.label_str)
                self.relate[len(self.type) - 1] = instance._label.label_str
            self.input_labels.append(instance._label.label_str)
            temp = []
            for i in xrange(instance._feature_vector.numFeatures()):
                temp.append(float(instance._feature_vector.get(i)))
            temp_toto.append(temp)
            self.inputlayer = len(temp)
            self.outlayer = len(self.type)
        self.mat = np.array(temp_toto)
        self.samples = len(self.mat)
        for label in self.input_labels:
            for i in xrange(0, len(self.relate)):
                if label == self.relate[i]:
                    self.input_label_categories.append(i)
                    break

        np.random.seed(1)
        W1 = ( 2 * np.random.randn(self.inputlayer, 20) / np.sqrt(self.inputlayer) ) -1
        b1 = np.zeros((1, 20))
        W2 = ( 2 * np.random.randn(20, self.outlayer) / np.sqrt(20) )-1
        b2 = np.zeros((1, self.outlayer))

        neuralnet = {}

        for i in xrange(0, 60000):

            # Forward propagation
            a1 = self.sigmoid(self.mat.dot(W1) + b1)
            exp_scores = np.exp((self.sigmoid( np.dot(self.mat,W1) + b1)).dot(W2) + b2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            # Backpropagation
            delt = probs
            delt[range(self.samples), self.input_label_categories] -= 1
            dW2 = np.dot(a1.T,delt)
            dW1 = np.dot(self.mat.T,  np.dot(delt,W2.T) * (1 - np.power(a1, 2)))

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.regularization * W2
            dW1 += self.regularization * W1

            # Gradient descent parameter update
            W1 = W1 - (self.alpha * dW1)
            b1 = b1 - (self.alpha * np.sum(np.dot(delt,W2.T) * (1 - np.power(a1, 2)), axis=0))
            W2 = W2 - (self.alpha * dW2)
            b2 = b2 - (self.alpha * np.sum(delt, axis=0))

            # Assign new parameters to the neuralnet
            # neuralnet =
        
        self.neuralnet = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        print "Training is Successful"

    def predict(self, instance):
        temp = []
        for i in range(instance._feature_vector.numFeatures()):
            temp.append(instance._feature_vector.feature_vec[i])
            #print temp
        return self.relate[self.result(np.array(temp))[0]]
