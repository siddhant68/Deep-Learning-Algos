# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generating Datasets
X, Y = make_blobs(n_samples=500, n_features=2, centers=2, random_state=0)

# Visualising Datasets
plt.style.use('seaborn')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Accent)
plt.show()

# Model and Helper Functions
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

# Implement Perceptron Learning Algorithm
# - Learn the weights
# - Reduce the loss
# - Make the predictions
    
def predict(X, weights):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return predictions

def loss(X, Y, weights):
    # Binary Cross Entropy
    Y_ = predict(X, weights)
    cost = np.mean(-Y*np.log(Y_) - (1-Y)*np.log(1-Y_))
    return cost

def update(X, Y, weights, learning_rate):
    # Make weights update for 1 epoch
    Y_ = predict(X, weights)
    dw = np.dot(X.T, Y_ - Y)
    
    m = X.shape[0]
    weights = weights - learning_rate * dw/(float(m))
    return weights
    
def train(X, Y, learning_rate=0.5, maxEpochs=100):
    # Modify input to handle bias term
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    
    # Init weights 0
    weights = np.zeros(X.shape[1])
    
    # Iterate over all epochs and make updates
    for epoch in range(maxEpochs):
        weights = update(X, Y, weights, learning_rate)
        if epoch % 10 == 0:
            l = loss(X, Y, weights)
            print('Epoch %d Loss %.4f'%(epoch, l))
                    
    return weights            

weights = train(X, Y)
    
    
    
    
    
    
