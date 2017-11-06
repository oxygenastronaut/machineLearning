# GD5firstClassifier.py

# Writing our own classifier from scratch

# essentially nearest neighbor

from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a, b)

class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train


	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)

		return predictions

	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		for i in range(1, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]


from sklearn import datasets
# import data
iris = datasets.load_iris()

X = iris.data
y = iris.target

# split data into test and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# for the sake of building my own classifier the next two lines will be commented
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()

# the goal of our classifier should be to get close to the preset one
# the desired accuracy is at least 90%

my_classifier = ScrappyKNN()


# train the classifier
my_classifier.fit(X_train, y_train)

# get predicitions based on the training
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)