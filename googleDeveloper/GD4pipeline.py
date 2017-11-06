# GD4pipeline.py

# first to code a basic pipeline

# then to understand how an algorithim learns from data

# lets build a spam classifier
# already have data and build a model

# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data 	# features are x
y = iris.target # labels are y
# that mimics the f(x) = y. That essentially treats the classifier
# as a function

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)
# test_size = 0.5 just means that half the data set is used for testing

# if we want to use a decision tree as our classifiers we can use
# the next two lines

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# but there's another way to do it
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

# those two methods are essentially the same except that they work
# a bit differently. They still provide similar results


my_classifier.fit(x_train, y_train)
# the above two lines creates a classifier and trains it to the data

predictions = my_classifier.predict(x_test)

# we have the predicted labels in predictions
# and the true labels within y_test
# to compare
from sklearn.metrics import accuracy_score
print accuracy_score(y_test,predictions)


# So what does it mean to learn a function from data?

