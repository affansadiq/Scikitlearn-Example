from sklearn import tree

features = [[140, 1],[130, 1],[150, 0],[170, 0]]
labels = ["apple", "apple", "orange", "orange"]

## Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

## it should predict oranges..
print clf.predict([[150,0]])