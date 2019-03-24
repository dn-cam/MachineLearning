import sklearn
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris(return_X_y=False)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df = pd.DataFrame(np.concatenate((iris.data, np.array([iris.target]).T), axis=1), 
                  columns=iris.feature_names + ['target']) 

df.head()

df.columns

def plotcolor(lst):
    colors = []
    for l in lst:
        if l==0:
            colors.append('red')
        else:
            colors.append('blue')
    return colors
colors=plotcolor(df.target)

plt.scatter(df['petal length (cm)'],df['petal width (cm)'], c=colors, label='Iris setosa') #Pass on the list created by the function here
plt.grid(True)
plt.title("Distribution of Classes for petal width and length")
plt.legend(loc = "upper left")
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                
        return self.weights
    
    
    
X = df[['petal length (cm)','petal width (cm)', ]]
Y = df[['target']]

percept = Perceptron(2)

for i in range(0, 150):
    if train_labels[i] == 0:
        train_labels[i] = 1
        
weights = percept.train(train, train_labels)
weights #50.01 , 277.614, 101.302

y_intercept = -101.302/50.01 #/101.302
x_intercept = -277.614/50.01 #/277.614

slope = y_intercept/(-1*x_intercept)

abline_values = [slope * i - y_intercept for i in df['petal length (cm)']]

decision = [(50.01 + x*101.302)/277.614 for x in df['petal length (cm)']]

plt.scatter(df['petal length (cm)'],df['petal width (cm)'], c=colors, label='Iris setosa')
plt.plot(df['petal length (cm)'], abline_values, 'b', c='g')
plt.grid(True)
plt.title("Distribution of Classes for petal width and length")
plt.legend(loc = "upper left")
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.ylim(0,2)
plt.show()

