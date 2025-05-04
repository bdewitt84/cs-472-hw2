#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
# Implemented by Makani Buckley and Brett DeWitt, 5/4/2025
#
import sys
import re
# Node class for the decision tree
from node import Split, Leaf
import math

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p

# A Bernoulli trial is an experiment that has two possible outcomes,
# a success and a failure.
# We denote the probability of success by p
# https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/distributions/bernoulli-distribution.html

# let p be the porportion of positive examples (makani)
def entropy(p):
    # >>>> YOUR CODE GOES HERE <<<<
    if p == 0 or p == 1:
        return 0
    else:
        p_pos = p
        p_neg = 1 - p
        return -(p_pos * math.log(p_pos, 2)) - (p_neg * math.log(p_neg, 2))


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurrences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of occurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<

    # First want overall entropy before the split
    e_before = entropy(py/total)

    # e1 is entropy of the subset where x_i=1
    # p is probability of class=1 when the feature is present
    if pxi == 0:
        # Avoid divide by zero if feature is not present in subset
        e1 = 0
    else:
        e1 = entropy(py_pxi / pxi)

    # e2 is entropy of the subset where x_i=0
    # p is probability of class=1 when the feature is NOT present
    if (total - pxi) == 0:
        # Avoid divide by zero if feature saturates data set
        e2 = 0
    else:
        e2 = entropy((py - py_pxi) / (total - pxi))

    # Get weighted average of e1 and e2
    w1 = pxi / total            # Number of occurrences of x_i=1 over total number of data points
    w2 = (total - pxi) / total  # Number of occurrences of x_i=0 over total number of data points

    e_after = (w1 * e1) + (w2 * e2)

    # Gain is calculated by (entropy before split) - (weighted average entropy after split)
    return (e_before - e_after)


# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable

# a function for calculating the parameters for infogain
# data is the data to count
# x is the feature to count by
def count_set (data, x):
    py_pxi = pxi = py = total = 0
    for val in data:
        if val[x] == 1 and val[-1] == 0:
            pxi += 1
        elif val[x] == 1 and val[-1] == 1:
            pxi += 1
            py += 1
            py_pxi += 1
        elif val[x] == 0 and val[-1] == 1:
            py += 1
        total += 1
    return py_pxi, pxi, py, total

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree_mak(data, varnames):
    # >>>> YOUR CODE GOES HERE <<<<
    # If all examples of S belong to the same class c
    class_comp = [sample[-1] for sample in data] # the classes in data
    c = class_comp[0]
    same = True
    for s in class_comp:
        if c != s:
            same = False
            break
        c = s
    if same: # if all samples are the same class
        # return a new leaf and label it with c
        print("PURE")
        return Leaf(varnames, c)
    else: # else
        print("TAINTED")
        # Select an attribute A maximizing information gain
        A = 0
        max_gain = 0
        for i in range(len(varnames)-1):
            counts = count_set(data, i)
            gain = infogain(*counts)
            print("%d: %f" %(i, gain))
            if gain > max_gain:
                max_gain = gain
                A = i
        print("Best Attribute: %s (%d): %f" % (varnames[A], A, max_gain))
        # Generate a new node DT with A as its test
        # For each value vi of A
            # Let Si = all examples in S with A = vi
            # Use ID3 to construct a decision tree DTi for Si
            # Make DTi a child of DT
        # Return DT
        return Leaf(varnames, 1)
    # For now, always return a leaf predicting "1":
    #return node.Leaf(varnames, 1)


def set_is_pure(data):
    label = data[0][-1]
    for sample in data:
        if sample[-1] != label:
            return False
    return True


def get_attribute_with_highest_info_gain(data):
    num_attributes = len(data[0])-1 # last item is class, not attribute, so subtract 1
    best_attribute = None
    max_gain = 0
    for attribute in range(num_attributes):
        gain = infogain(*count_set(data, attribute))
        if gain >= max_gain:
            max_gain = gain
            best_attribute = attribute
    return best_attribute


def partition_data_based_on_attribute(data, attribute_to_split):
    subset_without_attribute = [sample for sample in data if sample[attribute_to_split] == 0]
    subset_with_attribute = [sample for sample in data if sample[attribute_to_split] == 1]
    return subset_without_attribute, subset_with_attribute


def get_majority_class(data):
    count_label_0 = len([sample for sample in data if sample[-1] == 0])
    count_label_1 = len([sample for sample in data if sample[-1] == 1])
    if ((count_label_0 > count_label_1) or count_label_0 == count_label_1):
        return 0
    else:
        return 1


def build_tree(data, varnames):
    # If all samples have the same class:
    if set_is_pure(data):
        # return a Leaf node labeled with that class
        label = data[0][-1] # select label from arbitrary sample
        return Leaf(varnames, label)

    else:
        # Find the best attribute to split on (based on info gain)
        attribute_to_split = get_attribute_with_highest_info_gain(data)
        # Partition the data into:
        # left: samples where attribute == 0
        # right: samples where attribute == 1
        left_data, right_data = partition_data_based_on_attribute(data, attribute_to_split)
        # If either partition is empty:
        if not left_data or not right_data:
            # Return a Leaf node labeled with the majority class of current data
            return Leaf(varnames, get_majority_class(left_data))
        else:
            # Recursively call build_tree on left data → left child
            left_node = build_tree(left_data, varnames)
            # Recursively call build_tree on right data → right child
            right_node = build_tree(right_data, varnames)
            return Split(varnames, attribute_to_split, left_node, right_node)


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
