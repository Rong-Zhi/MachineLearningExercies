from sklearn import linear_model
from sklearn import preprocessing, cross_validation
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, y, test_size=0.2)

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
print(regr.coef_)

score = regr.score(X_test,y_test)
print("score:",score)
prediction = regr.predict(X_test)
print("prediction:",prediction)