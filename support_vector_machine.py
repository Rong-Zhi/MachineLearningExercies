# the objectvie of Support vector machine is to find the best splitting boundary
# between data. for 2-d, it's kind of like the best fit line that devides your dataset.
# which contains the widest margin between support vectors.(decision boundary)

import numpy as np
from sklearn import preprocessing,cross_validation, neighbors,svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# clf = neighbors.KNeighborsClassifier()
clf = svm.SVC()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Confidence: ", format(confidence))

example_measures = np.array([[4,2,1,1,1,2,3]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print("Prediction: ", prediction)


# svm works faster than knn via scikit-learn
