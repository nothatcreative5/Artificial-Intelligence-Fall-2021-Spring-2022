First you need to install the following libraries:
pandas version 1.1.1
numpy version 1.19.1
matplotlib version 3.5.0
sklearn version 1.0.1
graphviz version 0.19
gvgen https://github.com/stricaud/gvgen

Depending on the method with which you installed the graphviz library, you may have to delete the following code on line 12.
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

Each code mentions the dataset and the method with which it creates the decision tree.Run the one you want to see.
For changing the hyper-parameters you can change the max_depth limitation on line 18 or you can change the number of bins on line 174 for diabetes_gain and on line 170 for diabetes_gini.

If you don't want to see the decision tree after each run you can change the view parameter to FALSE on line 143 for diabetes_gini and on line 147 for diabetes_gain.
