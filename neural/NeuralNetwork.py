from Methods import *
import numpy as np


class NeuralNetwork(Predictor):
    input_batch = 0
    class_labels = []
    input_labels = []
    nn_input_dim = 2  
    nn_output_dim = 2  
    epsilon = 0.1 
    reg_lambda = 0.01  
    class_label_map = {}
    model = 0
    num_examples = 0  


    def __init__(self):
        self.num_examples = 0  # training set size
        self.nn_input_dim = 2  # input layer dimensionality
        self.nn_output_dim = 2  # output layer dimensionality
        self.epsilon = 0.002  # learning rate for gradient descent
        self.reg_lambda = 0.05  # regularization strength
        self.class_labels = []
        self.class_label_map = {}
        self.input_batch = 0
        self.input_labels = []
        self.model = 0
        self.input_label_categories = []

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


    def make_prediction(self, x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        val = np.argmax((np.exp((self.sigmoid(x.dot(W1) + b1)).dot(W2) + b2)) / np.sum((np.exp((self.sigmoid(x.dot(W1) + b1)).dot(W2) + b2)), axis=1, keepdims=True),axis=1)
        return val
    
    def train(self, instances):
        temp_toto = []
        for instance in instances:
            if instance.label.label_str not in self.class_labels:
                self.class_labels.append(instance.label.label_str)
                self.class_label_map[len(self.class_labels) - 1] = instance.label.label_str
            self.input_labels.append(instance.label.label_str)
            temp = []
            for i in xrange(0, len(instance.feature_vector.feature_vec)):
                temp.append(instance.feature_vector.feature_vec[i])
            temp_toto.append(temp)
            self.nn_input_dim = len(temp)
            self.nn_output_dim = len(self.class_labels)
        self.input_batch = np.array(temp_toto)
        self.num_examples = len(self.input_batch)
        for label in self.input_labels:
            for i in xrange(0, len(self.class_label_map)):
                if label == self.class_label_map[i]:
                    self.input_label_categories.append(i)
                    break

        np.random.seed(1)
        W1 = ( 2 * np.random.randn(self.nn_input_dim, 20) / np.sqrt(self.nn_input_dim) ) -1 
        b1 = np.zeros((1, 20))
        W2 = ( 2 * np.random.randn(20, self.nn_output_dim) / np.sqrt(20) )-1
        b2 = np.zeros((1, self.nn_output_dim))

        model = {}

        for i in xrange(0, 60000):

            # Forward propagation
            a1 = self.sigmoid(self.input_batch.dot(W1) + b1)
            exp_scores = np.exp((self.sigmoid(self.input_batch.dot(W1) + b1)).dot(W2) + b2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            # Backpropagation
            delt = probs
            delt[range(self.num_examples), self.input_label_categories] -= 1
            dW2 = (a1.T).dot(delt)
            db2 = np.sum(delt, axis=0, keepdims=True)
            dW1 = np.dot(self.input_batch.T, delt.dot(W2.T) * (1 - np.power(a1, 2)))
            db1 = np.sum(delt.dot(W2.T) * (1 - np.power(a1, 2)), axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2

            # Assign new parameters to the model
            # model = 
        
        self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        print "Training"

    def predict(self, instance):
        temp = []
        for i in xrange(0, len(instance.feature_vector.feature_vec)):
            temp.append(instance.feature_vector.feature_vec[i])
            print temp
return self.class_label_map[self.make_prediction(np.array(temp))[0]]
