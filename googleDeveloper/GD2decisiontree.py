# GD2decisiontree.py

# many types of classifiers
'''
- Artificial nueral net
- support vector machine
- etc
'''

# decision trees are easy to read and understand which makes them 
# useful in practice

# machine learning problem
# Iris - identifying flowers based on certain properties such as 
# length of pedals

# there are four features to describe each flower and 1 label each

# GOALS
'''
1. Import a dataset
2. Train a classifier
3. Predict label for a new flower
4. Visualize the tree
'''

# sample datasets at
# http://scikit-learn.org/stable/datasets/

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

'''
print iris.feature_names
print iris.target_names		# target is what we want/labels
print iris.data[0]
print iris.target[0]
'''

# that successfully imports the data

# we split up the data in testing data and training data
# test data "tests" the classifier's accuracy

test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#print test_target
#print clf.predict(test_data)

# the above trains the classifier

# now to visualize the tree

'''
# visualization code
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file = dot_data, 
			feature_names = iris.feature_names, class_names = iris.target_names,
			filled = True, rounded = True, impurity = False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
'''

# the above code should produce a pdf to show the tree
# but it doesnt work in py3 so I'll link the tree
# http://scikit-learn.org/stable/modulehttp://scikit-learn.org/stable/modules/tree.htmls/tree.html
# about half way down it is the tree chart

# each node asks a question on the top, if true: left, if false: right

print test_data[1], test_target[1]

# Choosing good features is one of your most important jobs
# the better the features, the better the tree that can be built

# to predict something
# clf.predict([the numbers to represent the features])

