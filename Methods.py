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
"""
The implementation of a (binary) tree node object, which is a label with a left and right child
"""
class TreeNode():    
    def __init__(self, lbl):
        self.label = lbl
        self.leftChild = None
        self.rightChild = None
        self.splitVal = None
    
    def getSplitVal(self):
        return self.splitVal
        
    def setSplitVal(self, val):
        self.splitVal = val
    
    def getLabel(self):
        return self.label 
        
    def getChildren(self):
        return self.leftChild, self.rightChild
        
    def leftSubtree(self):
        return self.leftChild
        
    def rightSubtree(self):
        return self.rightChild
        
    def setLeft(self, node):
        self.leftChild = node
    
    def setRight(self, node):
        self.rightChild = node
        
    def setLabel(self, lbl):
        self.label = lbl


"""
The Decision Tree implementation
"""
class DecisionTree(Predictor):
    def __init__(self, gain):
        self.gain_type = gain
        self.tree = TreeNode("root")
        self.goodPredictions = 0
        self.badPredictions = 0
        self.totalPredictions = 0
        self.preds = {}
        self.lblCounts = {}
        
      
    """ Determine if all the labels in a data set are the same """ 
    def allSameLabel(self, instances):
        allSame = True
        currLabel = instances[0]._label
            
        for i in instances:
            if i._label != currLabel:
                allSame = False
            currLabel = i._label
            
        return allSame
          
    
    """ Set all the attributes from the non-empty training data """
    def setAttributes(self, instances):
        numAtts = len(instances[0]._feature_vector.getVector())
        numLines = 0
        
        for i in instances:
            numLines += 1
        
        attributes = [[]] * numAtts
        
        # Fill each attribute list
        for a in range(numLines):
            i = instances[a]

            label = i._label.__str__()
            currFeatures = i._feature_vector
            count = 0
            
            # For each attribute in the current line of data, add it to its
            #  corresponding attribute list
            for j in range(numAtts):
                attributes[count].append([label, currFeatures.get(j)])
                count += 1
        
        return attributes
         
    
    """ Determine if there are no attributes in the current data set """
    def noAttributes(self, instances):
        for i in instances:
            vec = i._feature_vector
            
            for j in range(vec.numFeatures()):
                if vec.get(j) is not None:
                     return False
                     
        return True   
    
    
    """ Calculate the beginning entropy of a feature """
    def calcEBefore(self, attribute, instance):
        labelCounts = {}
        totalFeatures = 0
        entropy = 0
        
        # (Recall that a feature is a [label, feature_value] pair)
        for feature in attribute:
            if not labelCounts.has_key(feature[0]):
                labelCounts[feature[0]] = 1
            else:
                labelCounts[feature[0]] += 1
   
            totalFeatures += 1

        # Calculate the entropy using the frequency of each label
        for key in labelCounts:
            freq = labelCounts[key]
            entropy -= ((1.0 * freq) / (1.0 * totalFeatures)) * math.log(((1.0 * freq) / (1.0 * totalFeatures)), 2)            
        
        self.lblCounts = labelCounts
        
        return entropy, labelCounts
    
    
    """ Calculate the left partition entropy of a feature """
    def calcELeftAndRight(self, att, labelCounts):
        sum = 0
        count = 0
        totalFeatures = 0
        total = len(att)
        
        leftFeatureCounts = {}
        rightFeatureCounts = {}
        entropyLeft = 0
        entropyRight = 0

        for feature in att:
            #print feature[1]
        
            sum += feature[1]
            totalFeatures += 1
        
        # We've chosen the median of all features values of the attribute 
        #  to be the split value, which allows us to effectively turn numerical
        #  (or categorical) data into categorical data    
        splitVal = float(sum) / float(totalFeatures)
        
        numLeft = 0
        numRight = 0
        
        # For each value in an attribute list, decide if the value should be in the left 
        #  or right partition and then calculate the left and right branch entropies
        for feature in att:
            feature_label = feature[0]
            feature_value = feature[1]
            
            # The current feature value belongs in the left partition
            if feature_value <= splitVal:
                numLeft += 1

                for key in labelCounts:
                    if key == feature_label:
                        if not leftFeatureCounts.has_key(key):
                            leftFeatureCounts[feature_label] = 1
                        else:
                            leftFeatureCounts[feature_label] += 1
            else:
                # The current feature value belongs in the right partition
                numRight += 1
                
                for key in labelCounts:
                    if key == feature_label:
                        if not rightFeatureCounts.has_key(key):
                            rightFeatureCounts[feature_label] = 1
                        else:
                            rightFeatureCounts[feature_label] += 1
                
        # Calculate and return the left and right entropies
        for key in leftFeatureCounts:
            val = leftFeatureCounts[key]
            entropyLeft -= ((1.0 * val) / (1.0 * numLeft)) * math.log(((1.0 * val) / (1.0 * numLeft)), 2)
            
        for key in rightFeatureCounts:
            val = rightFeatureCounts[key]
            entropyRight -= ((1.0 * val) / (1.0 * numRight)) * math.log(((1.0 * val) / (1.0 * numRight)), 2)
        
        # Calculate the weights of the left and right partitions            
        wLeft = (1.0 * numLeft) / (1.0 * (numLeft + numRight))  
        wRight = (1.0 * numRight) / (1.0 * (numLeft + numRight))  
                    
        return entropyLeft, entropyRight, wLeft, wRight, splitVal 
        
    
    """ Calculate the pure information gain from the attributes """    
    def calcPureIGValue(self, att, instances):
        infoGain = 0
        
        # Calculate the Entropy_before of the attribute
        e_before, labelCounts = self.calcEBefore(att, instances)
        
        # Calculate the Entropy_leftPartition and Entropy_rightPartition of the attribute
        e_left, e_right, w_left, w_right, splitVal = self.calcELeftAndRight(att, labelCounts)

        # Calculate the Entropy_after of the attribute
        e_after = w_left * e_left + w_right * e_right

        # Calculate and return the information gain from the attribute
        infoGain = e_before - e_after

        return infoGain, splitVal 
    
    
    """ Turn the data into categorical data if it is not already, then calculate the optimal
         attribute to split by
    """
    def chooseAttribute(self, attributes, instances):
        maxIG = 0
        splitVal = 0
        currIG = 0
        count = 0
        best = None
        atts = attributes

        # Calculate the highest split value
        #for att in attributes:
        for i in range(len(atts)):
            currIG, currSplitVal = self.calcPureIGValue(atts[i], instances)

            if currIG > maxIG:
                maxIG = currIG
                splitVal = currSplitVal
                best = atts[i]
            
            count += 1
        
        return best, count - 1, splitVal
    
    
    """ Calculate the most commonly occuring label and return a node with that label """
    def mostCommonLabelNode(self, instances):
        freqs = {}
        
        # Get the frequency
        for i in instances:
            if not freqs.has_key(i._label):
                freqs[i._label] = 1
            else:
                freqs[i._label] += 1
                
        mostCommonLabel = freqs[max(freqs.values())]
        
        return TreeNode(mostCommonLabel)


    # < For testing >
    def printTree(self, node):
        if isinstance(node, TreeNode):
            print "Root: " + node.getLabel()
        else:
            print "None"
       
            
    """ Run the recursive decision tree training algorithm with the pure 
        information gain option 
    """
    def splitPIG(self, instances, attributes, root): 
        # 1. attribute <-- the best attribute for splitting the {examples} 
        #       Split based on the pure information gain
        # 2. For each value of attribute, create a new child node
        # 3. Split training {examples} into their corresponding subtrees
        # 4. For each child node:
        #       if the subset is pure (i.e. all of one category): return the root
        #       else: splitPIG(child node, {subset})
        #       Implement a basic reduced error pruning algorithm by changing
        #           non-pure root nodes to their most common label
        
        # 1
        best = self.chooseAttribute(attributes, instances)
        attribute = best[0]
        attributeIndex = best[1]
        splitVal = best[2]

        # 2
        root.setLabel(str(attributeIndex))
        root.setSplitVal(splitVal)
        leftInstances = []
        rightInstances = []

        # 3
        for instance in instances:
            vec = instance._feature_vector.getVector()

            if vec[attributeIndex] <= splitVal:
                leftInstances.append(instance)
            else:
                rightInstances.append(instance)
    
        # 4            
        if len(rightInstances) == 0:
            if len(leftInstances) != 0:
                root.setLabel(leftInstances[0]._label.__str__())
            return root 
            
        if self.allSameLabel(rightInstances):
            root.setLabel(rightInstances[0]._label.__str__())
            return root
        
        if len(leftInstances) == 0:
            if len(rightInstances) != 0:
                root.setLabel(rightInstances[0]._label.__str__())
            return root
            
        if self.allSameLabel(leftInstances):
            root.setLabel(leftInstances[0]._label.__str__())
            return root

        attSubset = attributes
        leftChild = TreeNode("Left")
        rightChild = TreeNode("Right")
        root.setLeft(leftChild)
        root.setRight(rightChild)
        
        if attSubset.count(attribute) > 0:     
            attSubset.remove(attribute)
            return self.splitPIG(rightInstances, attSubset, rightChild)
            return self.splitPIG(leftInstances, attSubset, leftChild)
        else:
            return root
    
    
    """ Train the decision tree training algorithm """
    def train(self, instances):
        # < for testing > 
        print "Training the Decision Tree algorithm"
        
        attributes = self.setAttributes(instances)
        
        # If there's no training data, just return an empty tree node
        if len(instances) == 0:
            return TreeNode("None")

        # If all the examples have the same label, just return a tree node with that label
        if self.allSameLabel(instances):
            return TreeNode(instances[0]._label)
            
        # If there are no attributes in the sample data, just return a tree node of the
        #  most commonly occurring label    
        if self.noAttributes(instances):
            return self.mostCommonLabelNode(instances)

        # Call the appropriate decision tree algorithm
        if self.gain_type == "pig":
            node = self.splitPIG(instances, attributes, self.tree)
            self.tree = node
            #self.printTree(node)
            return self
        else:
            node = self.splitIGR(instances, attributes, self.tree)
            self.tree = node
            #self.printTree(node)
            return self

    
    """ Calculate the range of values in an attribute """
    def calcRange(self, att):
        mini = 0
        maxi = 0
        
        for i in range(len(att)):
            if att[i][1] < mini:
                mini = att[i][1]
            elif att[i][1] > maxi:
                maxi = att[i][1]
    
        spread = maxi - mini
        
        return spread
    
    
    """ Calculate the information gain ratio """
    def calcIGR(self, att, instances):
        infoGain = 0
        infoGainRatio = 0
        spread = 0
        
        # Calculate the range in the data
        spread = self.calcRange(att)
        
        # Calculate the Entropy_before of the attribute
        e_before, labelCounts = self.calcEBefore(att, instances)
        
        # Calculate the Entropy_leftPartition and Entropy_rightPartition of the attribute
        e_left, e_right, w_left, w_right, splitVal = self.calcELeftAndRight(att, labelCounts)

        # Calculate the Entropy_after of the attribute
        e_after = w_left * e_left + w_right * e_right

        # Calculate and return the information gain from the attribute
        infoGain = e_before - e_after

        # Calculate the information gain ratio
        if spread == 0:
            infoGainRatio = 0
        else:
            infoGainRatio = float(infoGain) / float(spread)

        #print spread

        return infoGainRatio, splitVal 
    
    
    """ Choose the attribute to split on, based on the maximum information gain ratio """
    def chooseAttributeIGR(self, attributes, instances):
        maxIGR = 0
        splitVal = 0
        currIGR = 0
        count = 0
        best = None
        atts = attributes

        # Calculate the highest split value
        for i in range(len(atts)):
            currIGR, currSplitVal = self.calcIGR(atts[i], instances)

            if currIGR > maxIGR:
                maxIGR = currIGR
                splitVal = currSplitVal
                best = atts[i]
            
            count += 1
        
        return best, count - 1, splitVal


    """ Run the recursive decision tree training algorithm with the 
        information gain ratio option 
    """
    def splitIGR(self, instances, attributes, root): 
        # 1. attribute <-- the best attribute for splitting the {examples}
        #       Split based on the information gain ratio 
        # 2. For each value of attribute, create a new child node
        # 3. Split training {examples} into their corresponding subtrees
        # 4. For each child node:
        #       if the subset is pure (i.e. all of one category): return the root
        #       else: splitPIG(child node, {subset})
        #       Implement a basic reduced error pruning algorithm by changing
        #           non-pure root nodes to their most common label
        
        # 1
        best = self.chooseAttributeIGR(attributes, instances)
        attribute = best[0]
        attributeIndex = best[1]
        splitVal = best[2]

        #print splitVal

        # 2
        root.setLabel(str(attributeIndex))
        root.setSplitVal(splitVal)
        leftInstances = []
        rightInstances = []

        # 3
        for instance in instances:
            vec = instance._feature_vector.getVector()

            if vec[attributeIndex] <= splitVal:
                leftInstances.append(instance)
            else:
                rightInstances.append(instance)

        # 4            
        if len(rightInstances) == 0:   
            if len(leftInstances) != 0:
                root.setLabel(leftInstances[0]._label.__str__())   
            return root 
            
        if self.allSameLabel(rightInstances):
            root.setLabel(rightInstances[0]._label.__str__())
            return root
        
        if len(leftInstances) == 0:
            if len(rightInstances) != 0:
                root.setLabel(rightInstances[0]._label.__str__())   
            return root
            
        if self.allSameLabel(leftInstances):
            root.setLabel(leftInstances[0]._label.__str__())
            return root

        attSubset = attributes
        leftChild = TreeNode("Left")
        rightChild = TreeNode("Right")
        root.setLeft(leftChild)
        root.setRight(rightChild)
        
        if attSubset.count(attribute) > 0:  
            attSubset.remove(attribute)
            return self.splitPIG(rightInstances, attSubset, rightChild)
            return self.splitPIG(leftInstances, attSubset, leftChild)
        else:
            return root       
            
    
    """ Recursively traverse the training tree to make a prediction about the instance """
    def traverseTree(self, instance, root):
        # Get the test instance's label and features
        testLabel = instance.getLabel()
        testFeatures = instance.getVector()
        
        # Base case - once we've reached a node, return the label at that node
        if root.leftSubtree() is None and root.rightSubtree() is None:
            return root.getLabel
        
        # Otherwise, keep recursing until a leaf is encountered
        testIndex = int(testLabel)
        
        if testFeatures[testIndex] <= root.getSplitVal():
            return traverseTree(instance, root.leftSubtree())
        else:
            return traverseTree(instance, root.rightSubtree())
    
    
    # < for testing - display the metrics that are supposed to be calculated >
    def printMetrics(self, prediction):
        accuracy = 0.00
        precision = 0.00
        recall = 0.00
         
        if self.totalPredictions != 0:
            accuracy = float(self.goodPredictions) / float(self.totalPredictions)
            
            if self.preds.has_key(prediction):
                totalNumLabelGuesses = self.preds[prediction]
                precision = float(self.goodPredictions) / float(totalNumLabelGuesses)
            
            if self.lblCounts.has_key(prediction):
                totalNumLabel = self.lblCounts[prediction]
                recall = float(self.goodPredictions) / float(totalNumLabel)
            
        print "Accuracy: " + str(accuracy)
        print "Precision: " + str(precision)
        print "Recall: " + str(recall)
             
     
    """ Make a prediction about a test instance """               
    def predict(self, instance): 
        # < for testing >
        print "Making predictions with the Decision Tree algorithm"

        testLabel = instance.getLabel()
        prediction = self.traverseTree(instance, self.tree)

        # Keep track of how many times each label was predicted
        if not self.preds.has_key(prediction):
            self.preds[prediction] = 1
        else:
            self.preds[prediction] += 1

        # Track how many good, bad, and total predictions were made
        if testLabel == prediction:
            self.goodPredictions += 1
        else:
            self.badPredictions += 1
        
        self.totalPredictions += 1

        self.printMetrics(prediction)

        return prediction              

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
                mean = self.cal_mean(classes[label], feature)
                stdev = self.cal_stdev(classes[label], feature, mean)
                self.summaries[label][feature] = mean, stdev

    def cal_mean(self, instances, feature):  # Can be improved with np
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

class DecisionTree(Predictor):
        def __init__(self, summaries):
            pass
        def train(self, instances):
            pass
        def predict(self, instances):
            pass
class NeuralNetwork(Predictor):
    def __init__(self):
        self.samples = 0  # training set size
        self.inputsize = 0  # input layer dimensionality
        self.outputsize = 0 
        self.alpha = 0.002  
        self.regularization = 0.05
        self.classes = []
        self.relate = {}
        self.mat = 0
        self.labels = []
        self.neural_model = 0
        self.input_label = []
        self.epoch=2000

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def calc_output(self, x):
        W1, b1, W2, b2 = self.neural_model['W1'], self.neural_model['b1'], self.neural_model['W2'], self.neural_model['b2']
        val = np.argmax((np.exp((self.sigmoid(x.dot(W1) + b1)).dot(W2) + b2)) / 
                np.sum((np.exp((self.sigmoid(x.dot(W1) + b1)).dot(W2) + b2)),
                axis=1, keepdims=True),axis=1)
        return val
    
    def train(self, instances):
        temp_toto = []
        for instance in instances:
            label = str(instance._label.label_str)
            if label not in self.classes:
                self.classes.append(instance._label.label_str)
                self.relate[len(self.classes) - 1] = instance._label.label_str
            self.labels.append(instance._label.label_str)
            temp = []
            for i in xrange(instance._feature_vector.numFeatures()):
                temp.append(float(instance._feature_vector.get(i)))
            temp_toto.append(temp)
            self.inputsize = len(temp)
            self.outputsize = len(self.classes)
        self.mat = np.array(temp_toto)
        self.samples = len(self.mat)
        for label in self.labels:
            for i in xrange(0, len(self.relate)):
                if label == self.relate[i]:
                    self.input_label.append(i)
                    break

        np.random.seed(1)
        W1 = ( 2 * np.random.randn(self.inputsize, 20) / np.sqrt(self.inputsize) ) -1
        b1 = np.zeros((1, 20))
        W2 = ( 2 * np.random.randn(20, self.outputsize) / np.sqrt(20) )-1
        b2 = np.zeros((1, self.outputsize))

        neural_model = {}

        for i in range(1, self.epoch):

            # Forward propagation
            a1 = self.sigmoid(self.mat.dot(W1) + b1)
            exp_scores = np.exp((self.sigmoid( np.dot(self.mat,W1) + b1)).dot(W2) + b2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delt = probs
            delt[range(self.samples), self.input_label] -= 1
            dW2 = np.dot(a1.T,delt)
            dW1 = np.dot(self.mat.T,  np.dot(delt,W2.T) * (1 - np.power(a1, 2)))

            # regularization
            dW2 += self.regularization * W2
            dW1 += self.regularization * W1

            # Gradient descent update
            W1 = W1 - (self.alpha * dW1)
            b1 = b1 - (self.alpha * np.sum(np.dot(delt,W2.T) * (1 - np.power(a1, 2)), axis=0))
            W2 = W2 - (self.alpha * dW2)
            b2 = b2 - (self.alpha * np.sum(delt, axis=0))

        
        self.neural_model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def predict(self, instance):
        temp = []
        for i in range(instance._feature_vector.numFeatures()):
            temp.append(instance._feature_vector.feature_vec[i])
        return self.relate[self.calc_output(np.array(temp))[0]]

