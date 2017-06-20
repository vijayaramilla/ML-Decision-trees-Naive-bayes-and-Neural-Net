from Methods import *
import numpy as np


class NeuralNetwork(Predictor):
    input_batch = 0
    input_labels = []
    num_samples = 0  # training set size
    input_nodes_count = 2  # input layer dimensionality
    output_nodes_count = 2  # output layer dimensionality

    # Gradient descent parameters (I picked these by hand)
    learn_rate = 0.01  # learning rate for gradient descent
    reg_strength = 0.01  # regularization strength
    class_labels = []
    class_label_map = {}
    model = 0

    def __init__(self):
        self.num_samples = 0  # training set size
        self.input_nodes_count = 2  # input layer dimensionality
        self.output_nodes_count = 2  # output layer dimensionality
        self.learn_rate = 0.0001  # learning rate for gradient descent
        self.reg_strength = 0.01  # regularization strength
        self.class_labels = []
        self.class_label_map = {}
        self.input_batch = 0
        self.input_labels = []
        self.model = 0
        self.input_label_categories = []

    def build_model(self, hidden_nodes_count, epoch_count=10000):

        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(1234)
        hidden_1 = np.random.randn(self.input_nodes_count, hidden_nodes_count) / np.sqrt(self.input_nodes_count)
        hidden_1_bias = np.zeros((1, hidden_nodes_count))
        hidden_2 = np.random.randn(hidden_nodes_count, self.output_nodes_count) / np.sqrt(hidden_nodes_count)
        hidden_2_bias = np.zeros((1, self.output_nodes_count))

        for i in xrange(0, epoch_count):
            output_1, probs = self.forward_propagate(hidden_1, hidden_1_bias, hidden_2, hidden_2_bias)

            derivative_1, derivative_2, derivative_bias_1, derivative_bias_2 = self.back_propagate(hidden_2, output_1,
                                                                                                   probs)
            derivative_1 += self.reg_strength * hidden_2
            derivative_2 += self.reg_strength * hidden_1
            hidden_1, hidden_1_bias, hidden_2, hidden_2_bias = self.tune_weights(derivative_1, derivative_2,
                                                                                 derivative_bias_1, derivative_bias_2,
                                                                                 hidden_1, hidden_1_bias, hidden_2,
                                                                                 hidden_2_bias)
            self.model = {'weights_1': hidden_1, 'bias_1': hidden_1_bias, 'weights_2': hidden_2, 'bias_2': hidden_2_bias}
        print "Training Successful"

    def back_propagate(self, hidden_2, output_1, probs):
        # back
        error_3 = probs
        error_3[range(self.num_samples), self.input_label_categories] -= 1
        derivative_1 = (output_1.T).dot(error_3)
        derivative_bias_1 = np.sum(error_3, axis=0, keepdims=True)
        error2 = error_3.dot(hidden_2.T) * (1 - np.power(output_1, 2))
        derivative_2 = np.dot(self.input_batch.T, error2)
        derivative_bias_2 = np.sum(error2, axis=0)
        return derivative_1, derivative_2, derivative_bias_1, derivative_bias_2

    def forward_propagate(self, hidden_1, hidden_1_bias, hidden_2, hidden_2_bias):
        # forward
        output_1_pre_sigmoid = self.input_batch.dot(hidden_1) + hidden_1_bias
        output_1 = np.tanh(output_1_pre_sigmoid)
        output_2_pre_softmax = output_1.dot(hidden_2) + hidden_2_bias
        output_2 = np.exp(output_2_pre_softmax)
        probs = output_2 / np.sum(output_2, axis=1, keepdims=True)
        return output_1, probs

    def tune_weights(self, derivative_1, derivative_2, derivative_bias_1, derivative_bias_2, hidden_1, hidden_1_bias,
                     hidden_2, hidden_2_bias):
        # Gradient descent parameter update
        hidden_1 += -self.learn_rate * derivative_2
        hidden_1_bias += -self.learn_rate * derivative_bias_2
        hidden_2 += -self.learn_rate * derivative_1
        hidden_2_bias += -self.learn_rate * derivative_bias_1
        return hidden_1, hidden_1_bias, hidden_2, hidden_2_bias

    def train(self, instances):
        temp_toto = []
        for instance in instances:
            if instance._label.label_str not in self.class_labels:
                self.class_labels.append(instance._label.label_str)
                self.class_label_map[len(self.class_labels) - 1] = instance._label.label_str
            self.input_labels.append(instance._label.label_str)
            temp = []
            for i in range(instance._feature_vector.numFeatures()):
                #temp.append(instance.get_feature_vector()[i])
                temp.append(instance._feature_vector.get(i))
            temp_toto.append(temp)
            self.input_nodes_count = len(temp)
            self.output_nodes_count = len(self.class_labels)
        self.input_batch = np.array(temp_toto)
        self.num_samples = len(self.input_batch)
        for label in self.input_labels:
            for i in xrange(0, len(self.class_label_map)):
                if label == self.class_label_map[i]:
                    self.input_label_categories.append(i)
                    break
        self.build_model(self.input_nodes_count * self.output_nodes_count)

    def make_prediction(self, x):
        weights_1, bias_1, weights_2, bias_2 = self.model['weights_1'], self.model['bias_1'], self.model['weights_2'], self.model['bias_2']
        # Forward propagation
        z1 = x.dot(weights_1) + bias_1
        a1 = np.tanh(z1)
        z2 = a1.dot(weights_2) + bias_2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def predict(self, instance):
        temp = []
        for i in range(instance._feature_vector.numFeatures()):
            temp.append(instance._feature_vector.get(i))
        return self.class_label_map[self.make_prediction(np.array(temp))[0]]
