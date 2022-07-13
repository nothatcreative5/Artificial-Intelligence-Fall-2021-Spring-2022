import pandas as pd
import numpy as np
import gvgen
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from graphviz import Source

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

bins = []
HEADERS = []
y_pred = []
max_depth = 3
Correct_Guess = 0


# Decision tree
class Tree:
    def __init__(self, data, depth=0, parent_attribute=None):
        if parent_attribute is None:
            parent_attribute = []
        self.children = dict()
        self.attribute = -1
        self.depth = depth
        self.attribute_name = 'Leaf'
        self.gini = 2
        self.plus = 0
        self.negative = 0
        self.parent_attribute = parent_attribute
        self.data = np.array(data)

    def __str__(self):
        return self.attribute_name + "\n" + "Gini: " + str(self.gini) + "\n" + "[" + str(self.plus) + "/" + str(
            self.negative) + "]"

    def calculate_gini(self, gini, difference, attribute):
        total = 0
        for key, value in difference.items():
            total += len(value)
        f_gini = 0
        for key, value in gini.items():
            f_gini += value[0] * ((value[1] + value[2]) / total)
        if f_gini < self.gini:
            self.gini = f_gini
            self.attribute = attribute
            self.attribute_name = HEADERS[attribute]

    def train(self):
        self.plus = np.sum(self.data.T[-1])
        self.negative = self.data.shape[0] - self.plus
        self.attribute_name = str(int(self.plus > self.negative))

        self.gini = 1 - (self.plus / (self.plus + self.negative)) ** 2 - (
                self.negative / (self.plus + self.negative)) ** 2

        if self.depth == max_depth:
            return

        difference = dict()

        for i in range(len(self.data[0]) - 1):
            for row in self.data:
                if row[i] in difference:
                    difference[row[i]].append(row)
                else:
                    difference[row[i]] = []
                    difference[row[i]].append(row)
            self.choose_attribute(difference, i)
            if self.attribute == i:
                self.children.clear()
                for key, value in difference.items():
                    next_parent_attribute = self.parent_attribute.copy()
                    next_parent_attribute.append(self.attribute)
                    self.children[key] = Tree(value, depth=self.depth + 1, parent_attribute=next_parent_attribute)
            difference.clear()

        for key, value in self.children.items():
            value.train()

        if len(self.children) == 0:
            self.attribute_name = str(int(self.plus > self.negative))

    def choose_attribute(self, difference, attribute):
        gini = dict()
        for key, value in difference.items():
            gini[key] = []
            plus = 0
            negative = 0
            for each in value:
                if each[-1] == 1:
                    plus += 1
                else:
                    negative += 1
            gini[key].append(1 - (plus / (plus + negative)) ** 2 - (negative / (plus + negative)) ** 2)
            gini[key].append(plus)
            gini[key].append(negative)
        self.calculate_gini(gini, difference, attribute)


# function for discretizing our continuous data.
def discretize(data: np.array, number_of_bins):
    global bins
    for i in range(8):
        bins.append(pd.cut(data[i], number_of_bins, retbins=True)[1])
        data[i] = np.digitize(data[i], bins[i], right=True)


def get_edge_name(key, val: Tree):
    attribute_bin = bins[val.attribute]
    key = int(key)
    return str(round(attribute_bin[key - 1], 2)) + " <= x <= " + str(round(attribute_bin[key], 2))


def graphMaker(g, my_tree: Tree):
    if len(my_tree.children) == 0:
        myItem = g.newItem(my_tree.__str__())
        return myItem
    else:
        myItem = g.newItem(my_tree.__str__())
        for key, val in my_tree.children.items():
            newTree = graphMaker(g, val)
            link = g.newLink(myItem, newTree)
            g.propertyAppend(link, "color", "darkblue")
            g.propertyAppend(link, "label", get_edge_name(key, my_tree))
        return myItem


def makeVisualGraph(my_tree):
    g = gvgen.GvGen()
    graphMaker(g, my_tree)
    string = ""
    my_file = open("output_graphviz.txt", 'w')
    g.dot(my_file)
    my_file.close()
    my_file = open("output_graphviz.txt", 'r')
    lines = my_file.readlines()[1:]
    for line in lines:
        string = string + line
    src = Source(string)
    src.render(view=True)


def test(row: np.array, root):
    global Correct_Guess
    if len(root.children) == 0:
        y_pred.append(int(root.plus > root.negative))
        if int(root.plus > root.negative) == int(row[-1]):
            Correct_Guess += 1
        return
    if row[root.attribute] not in root.children.keys():
        if np.random.rand() > 0.5:
            y_pred.append(int(row[-1]))
            Correct_Guess += 1
        else:
            y_pred.append(1 - int(row[-1]))
        return
    test(row, root.children[row[root.attribute]])


if __name__ == '__main__':
    # read attributes from file.
    df = pd.read_csv('diabetes.csv')
    HEADERS = df.columns

    # Put all attributes into a numpy matrix.
    attributes = df.to_numpy().T
    discretize(data=attributes, number_of_bins=5)
    attributes_train, attributes_test = train_test_split(attributes.T, test_size=0.2, random_state=2)

    root = Tree(attributes_train)
    root.train()
    makeVisualGraph(root)

    for row in attributes_train:
        test(row, root)
    print("Accuracy for train data: " + str(round(Correct_Guess / len(attributes_train) * 100, 2)) + "%")
    Correct_Guess = 0

    cm = confusion_matrix(attributes_train.T[-1], y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    y_pred = []

    for row in attributes_test:
        test(row, root)
    print("Accuracy for test data: " + str(round(Correct_Guess / len(attributes_test) * 100, 2)) + "%")
    Correct_Guess = 0

    cm = confusion_matrix(attributes_test.T[-1], y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    average = 0
    for i in range(100):
        attributes_train, attributes_test = train_test_split(attributes.T, test_size=0.2)

        root = Tree(attributes_train)
        root.train()

        for row in attributes_test:
            test(row, root)
        average += round(Correct_Guess / len(attributes_test) * 100, 2)
        Correct_Guess = 0

    print("Average Accuracy of tree on test data is : " + str(round(average / 100, 2)) + "%")
