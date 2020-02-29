import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import arange


breast_cancer = sklearn.datasets.load_breast_cancer()
a = breast_cancer.data
b = breast_cancer.target


print(a.shape, b.shape)

data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['class'] = breast_cancer.target

# how many rows we have from each class
print(data['class'].value_counts())

data.groupby('class').mean()

a = data.drop('class', axis=1)
b = data['class']

# split data to test and train
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.1, stratify=b, random_state=1)

print(a_train.mean())
print("********************")
print(a_test.mean())

a_train = a_train.values
a_test = a_test.values


# Implementing Perceptron Class
class Perceptron:

    # initialize the weights vector w and threshold b to None
    def __init__(self):
        self.w = None
        self.b = None

    # takes input values x as an argument and perform the weighted aggregation of inputs
    # (dot product between w.x) and returns the value 1 if the aggregation is greater than the threshold b else 0
    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0

    # calculates the predicted outcome and returns a list of predictions
    def predict(self, a):
        output = []
        for x in a:
            result = self.model(x)
            output.append(result)
        return np.array(output)

    # learn the best possible weight vector w and threshold value b for the given data
    def fit(self, X, Y, epochs=1, lr=1):

        self.w = np.ones(a.shape[1])
        self.b = 0

        accuracy = {}
        max_accuracy = 0

        wt_matrix = []

        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1

            wt_matrix.append(self.w)

            pred = self.predict(X)
            accuracy[i] = accuracy_score(Y, pred)
            if accuracy[i] > max_accuracy:
                max_accuracy = accuracy[i]
                j = i
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb

        print("maximum accuracy is: ", max_accuracy, " in iteration: ", j)
        return np.array(wt_matrix)


accuracy = []
lr_values = np.arange(0.3, 3, 0.3)
for i in arange(0.3, 3, 0.3):
    perceptron = Perceptron()
    weight = perceptron.fit(a_train, b_train, 10000, i)
    b_pred_test = perceptron.predict(a_test)
    print("Accuracy Score for test data: ", accuracy_score(b_pred_test, b_test))
    accuracy.append(accuracy_score(b_pred_test, b_test))


plt.figure()
plt.plot(lr_values, accuracy, marker='o')
plt.xlabel('learning rate')
plt.ylabel('accuracy')
plt.show()

accuracy = []
epoch_values = np.arange(2000, 10000, 1000)
for i in range(2000, 10000, 1000):
    perceptron = Perceptron()
    weight = perceptron.fit(a_train, b_train, i, 0.3)
    b_pred_test = perceptron.predict(a_test)
    accuracy.append(accuracy_score(b_pred_test, b_test))

plt.figure()
plt.plot(epoch_values, accuracy, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
