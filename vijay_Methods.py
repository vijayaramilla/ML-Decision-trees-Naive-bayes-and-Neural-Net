import math
import numpy as np
from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        #self.label_num = int(label)
        self.label_str = str(label)
        pass
        
    def __str__(self):
        print self.label_str
        pass

# the feature vectors will be stored in dictionaries so that they can be sparse structures
class FeatureVector:
    def __init__(self):
        self.feature_vec = {}
        pass
        
    def add(self, index, value):
        self.feature_vec[index] = value
        pass
        
    def get(self, index):
        val = self.feature_vec[index]
        return val
    
    # Added to give the total number of features in the dictionary
    def numFeatures(self):
        return len(self.feature_vec)

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass


#TODO: you must implement additional data structures for
#the three algorithms specified in the hw4 PDF

#for example, if you want to define a data structure for the
#DecisionTree algorithm, you could write

#class DecisionTree(Predictor):
	# class code

#Remember that if you subclass the Predictor base class, you must
#include methods called train() and predict() in your subclasses

class NaiveBayes(Predictor):

    def __init__(self):
        self.summaries = {}
        self.prior = {}
        self.total = 0
        pass

    def mean(self, numbers):
	return sum(numbers)/float(len(numbers))
 
    def stdev(self, numbers):
	avg = self.mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)    

    def summarize(self, vectors):
    	summary = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*vectors)]
	return summary

    def train(self, instances):
        # separate the instances by class and create a map
        separated = {}
        self.total = len(instances)
        for instance in instances:
            #ins = instances[i]
            label = str(instance._label.label_str)
            if (label not in separated):
                separated[label] = []
                self.prior[label] = 0
            self.prior[label] += 1
            vector = list()

        features = instances[0]._feature_vector.feature_vec.keys()
        # Training model
        for label in self.priors:
            self.priors[label] = self.priors[label] / float(len(instances))
            for feature in features:
                mean = self._mean(classes[label], feature)
                stdev = self._variance(classes[label], feature, mean)
                self.likelihoods[label][feature] = mean, variance



        #for j in range(instance._feature_vector.numFeatures()):
        #        vector.append(float(instance._feature_vector.get(j)))
        #    separated[label].append(vector)
        # Calculate mean and standard deviation for the class seperated data
        #for classValue, vectors in separated.iteritems():
        #    self.summaries[classValue] = self.summarize(vectors)

    def calculateProbability(self, x, mean, stdev):
	#exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	#return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
        #p = 1/(np.sqrt(2*np.pi*stdev)) * np.exp((-(x-mean)**2)/(2*stdev))
        p = (1/np.sqrt(2*np.pi*variance)) * np.exp((-1/(2*variance))*(value - mean)**2
         
        #print p
        return p

    def predict(self, instance):
        probabilities = {}
	for classValue, classSummaries in self.summaries.iteritems():
            #caluclate prior probability
            p1 = self.prior[classValue]/float(self.total)
            probabilities[classValue] = p1
	    for i in range(len(classSummaries)):
	        mean, stdev = classSummaries[i]
                #vector = list()
                for j in range(instance._feature_vector.numFeatures()):
                    x = float(instance._feature_vector.get(j))
		    probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
        print probabilities
        bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
	    if bestLabel is None or probability > bestProb:
    	        bestProb = probability
		bestLabel = classValue
        return bestLabel

class DecisionTree(Predictor):
        def __init__(self, summaries):
            pass
        def train(self, instances):
            pass
        def predict(self, instances):
            pass
class NeuralNetwork(Predictor):
        def __init__(self, summaries):
            #Hyperparameters which are constants and define the neural network.
            self.inputSize = 2
            self.outputSize = 2
            self.hiddenLayerSize = 0
        
            #Synapses weights matrices.
            self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
            self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        def forward_propogate(self, X):
            #Propagate inputs though network
            self.z2 = np.dot(X, self.W1)
            self.a2 = self.sigmoid(self.z2)
            self.z3 = np.dot(self.a2, self.W2)
            initResult = self.sigmoid(self.z3)
            return yHat

        def sigmoid(self, z):
            return 1/(1+np.exp(-z))

        def train(self, instances):
            pass
        def predict(self, instances):
            pass


