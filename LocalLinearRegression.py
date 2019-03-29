import sklearn
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy import linalg

data = pd.read_csv("us_contagious_disease.csv")

def kernelWeights(x1, X, lambda_):
    weights = []
    for x in X:
        weights.append([stats.norm(0, lambda_).cdf(x1-x)])
    return weights

def localLinRegression(X, Y, lambda_):
    fit = []
    for x in X:
        W = np.diagflat(np.asarray(kernelWeights(x, X, lambda_)))
        B = np.column_stack((np.ones(len(X)), X))
        b = np.array([1,x])
        L = np.dot(np.dot(np.dot(b, linalg.inv(np.dot(B.T, np.dot(W, B)))), B.T), W)
        fit.append(sum(L*Y))
    return fit


measles = data[data['disease'] == 'Measles']
measles_short = measles #.sample(frac=0.1)


df = measles_short.groupby(['year']).sum()

train_y = df['count']
train_x = measles.year.unique()

#X = train_x.as_matrix()
Y = train_y.as_matrix()

plt.figure(figsize=(20,7))
plt.scatter(train_x, Y, alpha= 0.8, c='b')
plt.grid(True)
plt.title("Year-wise Measles occurrences")
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()


fit_1 = localLinRegression(train_x, Y,1)
fit_2 = localLinRegression(train_x, Y,2)
fit_5 = localLinRegression(train_x, Y,5)
fit_10 = localLinRegression(train_x, Y,10)
fit_20 = localLinRegression(train_x, Y,20)
fit_50 = localLinRegression(train_x, Y,50)

plt.figure(figsize=(13,8))
plt.scatter(train_x, Y)
plt.grid(True)
plt.plot(train_x, fit_1, c='r', label='Lambda = 1')
plt.plot(train_x, fit_2, c = 'b', label='Lambda = 2')
plt.plot(train_x, fit_5, c = 'g', label='Lambda = 5')
plt.plot(train_x, fit_10, c = 'pink', label='Lambda = 10')
plt.plot(train_x, fit_20, c = 'yellow', label='Lambda = 20')
plt.plot(train_x, fit_50, c = 'orange',label='Lambda = 50')
plt.legend(loc="upper right")
plt.show()



import sklearn as sk
