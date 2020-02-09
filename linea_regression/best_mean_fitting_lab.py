import numpy as np
import matplotlib.pyplot as plt

import theano

# Available Data
X = np.asarray([4, 9, 10, 14, 4, 7, 12, 22, 1, 3, 8, 11, 5, 6, 10, 11, 16, 13, 13, 10])
Y = np.asarray([390, 580, 650, 730, 410, 530, 600, 790, 350, 400, 590, 640, 450, 520, 690, 690, 770, 700, 730, 640])

mval = np.random.rand()
cval = np.random.rand()

# Variables for Lineal Regression Model
x = theano.tensor.vector('x')
y = theano.tensor.vector('y')
m = theano.shared(mval, name='m')
c = theano.shared(cval, name='c')

# Lineal Regression Model defined
yh = np.dot(x, m) + c

# Number of samples
n = X.shape[0]

cost = theano.tensor.sum(theano.tensor.pow(yh - y, 2)) / (2 * n)
gradm = theano.tensor.grad(cost, m)
gradc = theano.tensor.grad(cost, c)

# Gradient Descent Algorithm
mn = m - 0.01 * gradm
cn = c - 0.01 * gradc

train = theano.function([x, y], cost, updates=[(m, mn), (c, cn)])

test = theano.function([x], yh)

# Training 1
for i in range(500):
    costm = train(X, Y)
    print(costm)

# Testing
a = np.linspace(0, 15, 15)
b = test(a)

plt.scatter(X, Y)
plt.plot()

# Training 2
for i in range(500):
    costm = train(X, Y)
    print(costm)

b2 = test(a)
plt.plot(a, b2, color='red')


# best fitting line
def best_fitting(X, Y):
    mb = ((np.mean(X) * np.mean(Y)) - (np.mean(X * Y))) / ((np.mean(X) ** 2) - (np.mean(X ** 2)))
    cb = ((np.mean(Y)) - (mb * np.mean(X)))
    return mb, cb


mbnew, cbnew = best_fitting(X, Y)
yh = []

for i in X:
    yh.append(i * mbnew + cbnew)

plt.plot(X, yh, color='green')

plt.show()
