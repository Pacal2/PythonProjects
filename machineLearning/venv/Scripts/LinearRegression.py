import tensorflow
import pandas as pd
import numpy as np
import keras
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

''' For loop - finding the best accuracy and saving that model
best = 0
for _ in range(100000):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(X_train, y_train)
    acc = linear.score(X_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        print(f"New best: {acc}")
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
print(f"Final best: {best}")
'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Co: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(X_test)
for X in range(len(predictions)):
    print(predictions[X], X_test[X], y_test[X])

p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()