# GD1helloworld.py
from sklearn import tree

# in machine learning, measurements of training data are called
# features e.g. the weight of an orange when learning to compare 
# fruit or the texture of the fruit
# the label of the training data is essentially the name of the fruit

# for this program, the training data will be input manually

# for the features, 0 will represent bumpy fruit, 1 for smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 represents apples and 1 for oranges
labels = [0, 0, 1, 1]	
# scikit-learn uses real-valued features 

# step 2 is use these examples to train a classifier
# using a decision tree here
# essentially a box of rules

clf = tree.DecisionTreeClassifier()
# the decision tree essentially is a series of yes/no to determine qualities

# the training algorithim is included in scikit-learn, called fit
clf = clf.fit(features, labels)

print clf.predict([[150, 0]])	# 150g and bumpy
# once executed, the above should output a 1 for oranges. 
# the classifier predicted that the new example was an orange

# IMPORTANT CONCEPTS
'''
How does this work in the real world?
How much training data do you need?
How is the tree created?
What makes a good feature?
'''