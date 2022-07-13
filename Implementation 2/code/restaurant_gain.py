import pandas as pd
import numpy as np
import gvgen
import os
import warnings

from graphviz import Source

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
warnings.filterwarnings('ignore')


bins = []
HEADERS = []
max_depth = 10
Correct_Guess = 0


# Decision tree
class Tree:
    def __init__(self, data: np.array, parent_attribute=None, depth=0):
        if parent_attribute is None:
            parent_attribute = []
        self.children = dict()
        self.attribute = -1
        self.depth = depth
        self.attribute_name = 'Leaf'
        self.information_gain = -1
        self.entropy = -1
        self.plus = 0
        self.negative = 0
        self.parent_attribute = parent_attribute
        self.data = np.array(data)

    def __str__(self):
        if len(self.children) == 0:
            return self.attribute_name + "\n" + "Entropy: " + str(self.entropy) + "\n" + "[" + str(
                self.plus) + "/" + str(
                self.negative) + "]"
        return self.attribute_name + "\n" + "Gain: " + str(self.information_gain) + "\n" + "[" + str(
            self.plus) + "/" + str(
            self.negative) + "]"

    def train(self):
        self.plus = np.sum(self.data.T[-1])
        self.negative = self.data.shape[0] - self.plus
        self.attribute_name = str(int(self.plus > self.negative))
        self.entropy = calculate_entropy(self.plus, self.negative)

        if self.depth == max_depth or self.plus == 0 or self.negative == 0:
            return

        difference = dict()

        for i in range(len(self.data[0]) - 1):
            if i in self.parent_attribute:
                continue
            for row in self.data:
                if row[i] in difference:
                    difference[row[i]].append(row)
                else:
                    difference[row[i]] = []
                    difference[row[i]].append(row)
            self.choose_attribute(difference, i)
            if self.attribute == i:
                self.children.clear()
                next_parent_attribute = self.parent_attribute.copy()
                next_parent_attribute.append(self.attribute)
                for key, value in difference.items():
                    self.children[key] = Tree(value, depth=self.depth + 1, parent_attribute=next_parent_attribute)
            difference.clear()

        for key, value in self.children.items():
            value.train()

        if len(self.children) == 0:
            self.attribute_name = str(int(self.plus > self.negative))

    def choose_attribute(self, difference, attribute):
        remainder_A = 0
        for key, value in difference.items():
            plus = 0
            negative = 0
            for each in value:
                if each[-1] == 1:
                    plus += 1
                else:
                    negative += 1
            remainder_A += ((plus + negative) / (self.plus + self.negative)) * calculate_entropy(plus, negative)
        if self.information_gain < self.entropy - remainder_A:
            self.information_gain = self.entropy - remainder_A
            self.attribute = attribute
            self.attribute_name = HEADERS[attribute]


# function for discretizing our continuous data.
def discretize(data: np.array, number_of_bins):
    global bins
    for i in range(8):
        bins.append(pd.cut(data[i], number_of_bins, retbins=True)[1])
        data[i] = np.digitize(data[i], bins[i], right=True)


def calculate_entropy(plus, negative):
    q = plus / (plus + negative)
    if q == 0 or q == 1:
        return 0
    return -(q * np.log2(q) + (1 - q)
             * np.log2(1 - q))


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
            g.propertyAppend(link, "label", key)
            # g.propertyAppend(link, "label", get_edge_name(key, my_tree))
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
        if int(root.plus > root.negative) == int(row[-1]):
            Correct_Guess += 1
        return
    if row[root.attribute] not in root.children.keys():
        if np.random.rand() > 0.5:
            Correct_Guess += 1
        return
    test(row, root.children[row[root.attribute]])


if __name__ == '__main__':
    df = pd.read_csv('restaurant.csv')
    HEADERS = df.columns
    attributes = df.to_numpy().T
    root = Tree(attributes.T)
    root.train()
    makeVisualGraph(root)
