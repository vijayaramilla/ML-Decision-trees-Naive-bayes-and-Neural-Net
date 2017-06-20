from Methods import *
import numpy as np
from collections import Counter, defaultdict
import operator
import math
import copy


class DecisionTree(Predictor):
    def __init__(self):
        self.method = 0
        self.tfringe = {}
        self.features_mat = []
        self.labels = []
        self.test_mat = []
        self.test_labels = []

    def ig_kernel_sum(self, veclen, toto, attr):
        for g, h in attr.iteritems():
            s_val = sum(h.values())
            toto += ((s_val * self.random_val(h)) / veclen)
        return toto

    def ig_kernel(self, dest, set, cur_attr):
        self.ig_ratio_kernel(dest, set, cur_attr)
        summation = 0
        return summation

    def get_ig(self, set, dest, parent):
        cur_attr = defaultdict(dict)
        length = len(set)
        summation = self.ig_kernel(dest, set, cur_attr)
        summation = self.ig_kernel_sum(length, summation, cur_attr)
        return parent - summation

    def get_ig_ratio(self, set, dest, parent):
        cur_attr = defaultdict(dict)
        length = len(set)
        self.ig_ratio_kernel(dest, set, cur_attr)
        summation = 0
        range_data = np.asarray(set, dtype=float)
        ranged = float(np.max(range_data) - np.min(range_data))
        if ranged != 0:
            return (parent - summation)/ranged
        else:
            return (parent - summation)/1

    @staticmethod
    def ig_ratio_kernel(dest, set, cur_attr):
        for i, j in zip(set, dest):
            if j in cur_attr[i]:
                cur_attr[i][j] += 1
            else:
                cur_attr[i][j] = 1

    def calculate_loss(self, set, attributes, val):
        count = Counter()
        check = 0
        for i in val:
            count[i] += 1
        temp = self.random_val(count)
        gain = {}
        for a in attributes:
            sample = np.copy(set[:, a])
            if self.method == 0:
                gain[a] = self.get_ig(sample, val, temp)
            else:
                gain[a] = self.get_ig_ratio(sample, val, temp)
        for k in gain:
            if gain[k] < 0.04:
                check += 1
        if check > (set.shape[0] / 2):
            return False
        else:
            return True

    def loss_min(self, set, attributes, val):
        count = Counter()
        for i in val:
            count[i] += 1
        temp = self.random_val(count)
        gain = {}
        for attr in attributes:
            sample = np.copy(set[:, attr])
            if self.method == 0:
                gain[attr] = self.get_ig(sample, val, temp)
            else:
                gain[attr] = self.get_ig_ratio(sample, val, temp)
        return max(gain.iteritems(), key=operator.itemgetter(1))[0]

    @staticmethod
    def return_highest(target):
        return Counter(target).most_common(1)[0][0]

    @staticmethod
    def ceiling(x):
        frac = x - math.floor(x)
        if frac >= 0.5:
            return math.ceil(x)
        else:
            return math.floor(x)

    @staticmethod
    def random_val(cnt):
        total = sum(cnt.values())
        add = .0
        for k, value in cnt.iteritems():
            temp = float(value) / float(total)
            add += -(temp * math.log(temp, 2))
        return add

    @staticmethod
    def get_Attribute(set, value_in, attribute):
        return set[set[:, attribute] == value_in]

    def make_vals(self, set, attribute, params):
        target = set[:, 0]
        val = np.unique(target)
        max_label = self.return_highest(target)
        return self.get_attr_vals(attribute, max_label, params, set, target, val)

    def get_attr_vals(self, attribute, max_label, params, set, target, val):
        if len(val) == 1:
            return val[0]
        elif len(attribute) - 1 == 0:
            return max_label
        else:
            root = self.loss_min(set, attribute, target)
            tree = {root: {}}
            for val in np.unique(set[:, root]):
                sample = self.get_Attribute(set, val, root)
                attribute = [attr for attr in attribute if attr != root]
                child = self.make_vals(sample, attribute, params)
                tree[root][val] = child
            return tree

    def prune(self, set, attribute, params):
        target = set[:, 0]
        u_vals = np.unique(target)
        max_label = self.return_highest(target)
        if len(u_vals) == 1:
            return u_vals[0]
        elif len(attribute) - 1 == 0:
            return max_label
        else:
            if self.calculate_loss(set, attribute, target):
                root = self.loss_min(set, attribute, target)
                tree = {root: {}}
                for val in np.unique(set[:, root]):
                    subsamples = self.get_Attribute(set, val, root)
                    subattrs = [attr for attr in attribute if attr != root]
                    child = self.prune(subsamples, subattrs, params)
                    tree[root][val] = child
                return tree

    def make_prediction(self, data):
        terminal = copy.deepcopy(self.tfringe)
        while isinstance(terminal, dict):
            if len(terminal.keys()) == 1:
                attr = terminal.keys()[0]
                uni = np.unique(np.asarray(self.labels))
                rand = np.random.randint(len(uni))
                try:
                    terminal = terminal[attr][data[attr]]
                except KeyError:
                    try:
                        terminal = uni[rand]
                    except KeyError:
                        terminal = uni[rand]
                    continue
            else:
                raise "KeyError"
        return terminal

    @staticmethod
    def precision_calc_tree(predictions, data):
        total, y = data.shape
        precision = {}
        recall = {}
        correct = 0
        truelabels = data.T[0]
        true_counts = Counter(truelabels)
        pred_counts = Counter(predictions)
        correct_label_counts = {}
        for i in true_counts:
            correct_label_counts[i] = 0
            for j in range(len(predictions)):
                if predictions[j] == truelabels[j] and predictions[j] == i:
                    correct_label_counts[i] += 1
        for i in correct_label_counts:
            precision[i] = float(correct_label_counts[i])/float(pred_counts[i])
            recall[i] = float(correct_label_counts[i])/float(true_counts[i])
        for i in range(len(truelabels)):
            if truelabels[i] == predictions[i]:
                correct += 1
        accuracy = correct/float(total)
        return correct, accuracy, precision, recall

    def train(self, instances):
        for i in instances:
            temp = []
            for j in range(len(i.get_feature_vector().keys())):
                temp.append(i.get_feature_vector()[j])
            self.features_mat.append(temp)
            self.labels.append(i.get_label())
        data = np.asarray(self.features_mat, dtype=np.float64)
        labels = np.asarray(self.labels)
        data = np.vstack((labels, data.T)).T
        x, y = data.shape
        attrs = range(data.shape[1])
        attrs.pop(0)
        for i in range(0, x):
            for j in range(1, y):
                data[i][j] = self.ceiling(float(data[i][j]))
        self.tfringe = self.make_vals(data, attrs, {})
        return self

    def marshall_input(self, instances):
        for i in instances:
            temp = []
            for j in range(len(i.get_feature_vector().keys())):
                temp.append(i.get_feature_vector()[j])
            self.test_mat.append(temp)
            self.test_labels.append(i.get_label())

    def predict(self, instances, cindex=0):
        self.marshall_input(instances)
        data = np.asarray(self.test_mat, dtype=np.float64)
        labels = np.asarray(self.test_labels)
        data = np.vstack((labels, data.T)).T
        prediction = []
        x, y = data.shape
        for i in range(0, x):
            for j in range(1, y):
                data[i][j] = self.ceiling(float(data[i][j]))
        for d in data:
            prediction.append(self.make_prediction(d))
        correct, accuracy, precision, recall = self.precision_calc_tree(prediction, data)
        print prediction
