import numpy as np
import math

N = 1000

np.random.seed(1)

TX = (np.random.rand(N, 2) * 1000).astype(np.int32) + 1
TY = (TX.min(axis=1) / TX.max(axis=1) <= 0.2).astype(np.int32)[np.newaxis].T

MU = TX.mean(axis=0)
SIGMA = TX.std(axis=0)

def standardize(X):
    return (X - MU) / SIGMA

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

TX = standardize(TX)

W1 = np.random.randn(2, 2) 
W2 = np.random.randn(2, 2) 
W3 = np.random.randn(1, 2) 
b1 = np.random.randn(2)    
b2 = np.random.randn(2)    
b3 = np.random.randn(1)    

def forward(X0):
    Z1 = np.dot(X0, W1.T) + b1
    X1 = sigmoid(Z1)
    Z2 = np.dot(X1, W2.T) + b2
    X2 = sigmoid(Z2)
    Z3 = np.dot(X2, W3.T) + b3
    X3 = sigmoid(Z3)

    return Z1, X1, Z2, X2, Z3, X3

def dsigmoid(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def delta_output(Z, Y):
    return (sigmoid(Z) - Y) * dsigmoid(Z)

def delta_hidden(Z, D, W):
    return dsigmoid(Z) * np.dot(D, W)

def backward(Y, Z3, Z2, Z1):
    D3 = delta_output(Z3, Y)
    D2 = delta_hidden(Z2, D3, W3)
    D1 = delta_hidden(Z1, D2, W2)

    return D3, D2, D1

ETA = 0.001

def dweight(D, X):
    return np.dot(D.T, X)

def dbias(D):
    return D.sum(axis=0)

def update_parameters(D3, X2, D2, X1, D1, X0):
    global W3, W2, W1, b3, b2, b1

    W3 = W3 - ETA * dweight(D3, X2)
    W2 = W2 - ETA * dweight(D2, X1)
    W1 = W1 - ETA * dweight(D1, X0)

    b3 = b3 - ETA * dbias(D3)
    b2 = b2 - ETA * dbias(D2)
    b1 = b1 - ETA * dbias(D1)

def train(X, Y):
    Z1, X1, Z2, X2, Z3, X3 = forward(X)

    D3, D2, D1 = backward(Y, Z3, Z2, Z1)

    update_parameters(D3, X2, D2, X1, D1, X)

EPOCH = 3000

def predict(X):
    return forward(X)[-1]

def E(Y, X):
    return 0.5 * ((Y - predict(X)) ** 2).sum()

BATCH = 100

for epoch in range(1, EPOCH + 1):
    p = np.random.permutation(len(TX))

    for i in range(math.ceil(len(TX) / BATCH)):
        indice = p[i*BATCH:(i+1)*BATCH]
        X0 = TX[indice]
        Y  = TY[indice]

        train(X0, Y)

    if epoch % 1000 == 0:
        print(f"epoch={epoch} {E(TY, TX)}")

def classify(X):
    return (predict(X) > 0.8).astype(np.int32)

TEST_N = 1000
testX = (np.random.rand(TEST_N, 2) * 1000).astype(np.int32) + 1
testY = (testX.min(axis=1) / testX.max(axis=1) <= 0.2).astype(np.int32)[np.newaxis].T

accuracy = (classify(standardize(testX)) == testY).sum() / TEST_N
