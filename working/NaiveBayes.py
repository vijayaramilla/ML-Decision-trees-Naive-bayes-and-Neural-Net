import numpy as np
from Methods import Predictor


class NaiveBayes(Predictor):

    def __init__(self):
        self.summaries = {}
        self.class_priors = {}

    def train(self, instances):
        classes = {}
        # Seperate the instances in to classes
        for instance in instances:
            label = str(instance._label.label_str)
            if label not in self.class_priors:
                self.class_priors[label] = 0
                self.summaries[label] = {}
                classes[label] = []
            classes[label].append(instance)
            self.class_priors[label] += 1
        features = instances[0]._feature_vector.feature_vec.keys()

        # Training
        for label in self.class_priors:
            self.class_priors[label] = self.class_priors[label] / float(len(instances))
            for feature in features:
                mean = self.mean(classes[label], feature)
                stdev = self.cal_stdev(classes[label], feature, mean)
                self.summaries[label][feature] = mean, stdev

    def calc_mean(self, instances, feature):  # Can be improved with np
        s = 0
        for instance in instances:
            s += instance._feature_vector.feature_vec[feature]
        return (1/float(len(instances))) * s

    def cal_stdev(self, instances, feature, mean):  # Can be improved with np
        sl = 0
        for instance in instances:
            sl += (instance._feature_vector.feature_vec[feature] - mean) ** 2
        return (1/float(len(instances) - 1)) * sl

    def calculateProbability(self, x, m):
        mean, stdev = m
        p = (1/np.sqrt(2*np.pi*stdev)) * np.exp((-1/(2*stdev))*(x - mean)**2)
        return p

    def predict(self, instance):
        posteriors = {}
        features = instance._feature_vector.feature_vec.keys()
        for label in self.class_priors:
            posteriors[label] = self.class_priors[label]
            for feature in features:
                posteriors[label] *= self.calculateProbability(instance._feature_vector.feature_vec[feature], self.summaries[label][feature])
        return max(posteriors, key=posteriors.get)
