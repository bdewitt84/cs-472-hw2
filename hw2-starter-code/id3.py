# !/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
import node
import math

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p

# let p be the porportion of positive examples (makani)
def entropy(p):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":
    """
    if len(p) == 0:
        raise ValueError("p is empty")
    p_pos = 0
    p_neg = 0

    for val in p:
        if val:
            p_pos += 1
        else:
            p_neg += 1    

    total = p_pos + p_neg
    """
    p_pos = p
    p_neg = 1 - p
    # let a + b = entropy
    a = 0 if p_pos == 0 else (-p_pos * math.log(p_pos, 2))
    b = 0 if p_neg == 0 else (-p_neg * math.log(p_neg, 2))
    return a + b


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurrences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of occurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<

    # First want total entropy before the split
    total_e = entropy(py/total)

    # e1 is entropy of
    # number of times feature occurs where class = 1
    # over number of times feature occurs
    # This gives us our probability of class=1 when the feature is present
    e1 = entropy(py_pxi / pxi)

    # e2 is entropy of
    # number of time feature DOES NOT occur where class = 1
    # over number of times feature DOES NOT occur
    # This gives us our probability of class=1 when the feature is NOT present
    e2 = entropy((py - py_pxi) / (total - pxi))

    # Get weighted average of e1 and e2

    weight1 = pxi / total           # Number of occurrences of x_i=1 over total number of data points
    weight2 = (total - pxi) / total   # Number of occurrences of x_i=0 over total number of data points

    weighted_avg = (weight1 * e1) + (weight2 * e2)

    # Gain is calculated by (total entropy) - (weighted average)
    return (total_e - weighted_avg)


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

# a function for splitting a set of data on a feature x
# returns a tuple of the subset of data with x=0 and the subset of data with x=1
def split_on(data, x):
    xpos = [s for s in data if s[x] == 1]
    xneg = [s for s in data if s[x] == 0]
    return xneg, xpos

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
def build_tree(data, varnames):
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
    if same == 0: # if all samples are the same class
        # return a new leaf and label it with c
        print("PURE")
        return node.Leaf(varnames, c)
    else:
    # Else
        # Select an attribute A maximizing information gain
        # Generate a new node DT with A as its test
        # For each value vi of A
            # Let Si = all examples in S with A = vi
            # Use ID3 to construct a decision tree DTi for Si
            # Make DTi a child of DT
        # Return DT
        print("TAINTED")
        return node.Leaf(varnames, 1)
    # For now, always return a leaf predicting "1":
    #return node.Leaf(varnames, 1)


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
