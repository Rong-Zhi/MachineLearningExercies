# K nearest neighbors
# have problems of ValueError: query data dimension
# must match training data dimension
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)  # id info is useless

X = np.array(df.drop(['class'], 1))     # features
y = np.array(df['class'])   # labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# define the classifier
clf = neighbors.KNeighborsClassifier()

# train the classifier
clf.fit(X_train, y_train)

# test
accuracy = clf.score(X_test, y_test)
print(accuracy)
# in percentage
print("Accuracey: {0:.1%}".format(accuracy))


example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)
prediction = clf.predict(example_measures)
print(prediction)