"""
Jeremy Bauchwitz
AI Assignment 4
"""

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
        return self.label_str
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

"""
TODO: you must implement additional data structures for
the three algorithms specified in the hw4 PDF

for example, if you want to define a data structure for the 
DecisionTree algorithm, you could write

class DecisionTree(Predictor):
	# class code

Remember that if you subclass the Predictor base class, you must
include methods called train() and predict() in your subclasses
"""

"""
The Decision Tree implementation
"""
class DecisionTree(Predictor):
    def __init__(self, gain):
        self.gain_type = gain
     
        
    def calcPureIGValue(self, instances):
        return 0    
      
        
    def train(self, instances): 
        print "Training the Decision Tree algorithm"
        print "Printing the labels from the current training data:"
        
        # <for testing>: print out all the features for each label
        for i in instances:
            print i._label,
            vec = i._feature_vector
            for j in range(vec.numFeatures()):
                print vec.get(j),
            print
        
        
        """ Begin the decision tree implementation here
        """
        
        # Run the decision tree training with the pure information gain option    
        if self.gain_type == "pig":
            # If there's no training data, just return an empty list
            if len(instances) == 0:
                return []

            # If all the examples have the same label, just 
            allSame = True
            currLabel = instances[0]._label
            
            for i in instances:
                if i._label != currLabel:
                    allSame = False
                currLabel = i._label
            
            # We now know that all the examples have the same label    
            if allSame:
                pass
            
            
            
            
             
            # Initialize the lists that will hold the nodes in the left and right branches    
            leftSubtree = list();
            rightSubtree = list();
        
            splitVal = self.calcPureIGValue(instances)
        
            for instance in instances:
                pass
            
            print "Successfully read the data"
        
        
        else:
            # Run the decision tree training with the information gain ratio option  
            pass    
            
            
            
                   
    def predict(self, instance): 
        print "Making predictions with the Naive Bayes algorithm"

	
"""
The Naive Bayes implementation
"""	
class NaiveBayes(Predictor):
    def train(self, instances): 
        print "Training the Naive Bayes algorithm"

    def predict(self, instance): 
        print "Making predictions with the Naive Bayes algorithm"
	

"""
The Neural Network implementation
"""	
class NeuralNetwork(Predictor):
    def train(self, instances): 
        print "Training the Neural Network algorithm"

    def predict(self, instance): 
        print "Making predictions with the Naive Bayes algorithm"