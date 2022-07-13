import pandas as pd
import numpy as np
import gvgen
import os

from graphviz import Source

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

HEADERS = []


class Tree:
    def __init__(self, data):
        self.children = dict()
        self.attribute = -1
        self.attribute_name = 'Leaf'
        self.gini = 2
        self.plus = 0
        self.negative = 0
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
        self.gini = 1 - (self.plus / (self.plus + self.negative)) ** 2 - (
                self.negative / (self.plus + self.negative)) ** 2
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
                    self.children[key] = Tree(value)
            difference.clear()
        for key, value in self.children.items():
            value.train()
        if self.attribute_name == 'Leaf' and self.plus > self.negative:
            self.attribute_name = '1'
        elif self.attribute_name == 'Leaf' and self.negative > self.plus:
            self.attribute_name = '0'

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


def graphMaker(g, my_tree):
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


if __name__ == "__main__":
    df = pd.read_csv('restaurant.csv')
    HEADERS = df.columns
    root = Tree(df.to_numpy())
    root.train()
    makeVisualGraph(root)
