import random
from scipy.spatial import distance

def euc(a,b):
	return distance.euclidean(a,b)

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
		best_index = 0
		for i in range (1, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]



from sklearn import datasets, svm
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .02)

## Decision Tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

## Nearest K-Neighbors
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict([[6.9,2.4,1.6,1.6],[7.4,3.5,2.8,2.5], [6.3,3.4,2.9,3.4]])
print predictions


import pydotplus
dot_data = tree.export_graphviz(my_classifier, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("pipeline.pdf")

from sklearn.metrics import accuracy_score

#print "  "
#print "Our Accuracy is : "
print accuracy_score(y_test, predictions)
#print predict