# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

# Generating Datasets
# make_blobs is linear and make_moons is non linear
X, Y = make_blobs(n_samples=500, n_features=2, centers=2, random_state=11)
# X, Y = make_moons(n_samples=1000, shuffle=True, noise=0.2, random_state=1)

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
    
def train(X, Y, learning_rate=0.8, maxEpochs=1000):
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

# Function to make predictions for X_test 
def getPredictions(X_test, weights, labels=True):
    if X_test.shape[1] != weights.shape[0]:
        ones = np.ones((X_test.shape[0], 1))
        X_test = np.hstack((ones, X_test))
    
    probs = predict(X_test, weights)
    
    if not labels:
        return probs
    else:
        labels = np.zeros(probs.shape)
        labels[probs >= 0.5] = 1
        return labels
    
# Visualising Hypothesis 

# Generating x1
x1 = np.linspace(-12, 2, 10)
print(x1)

# Calculating x2 using x1
x2 = -(weights[0] + weights[1]*x1)/weights[2]
print(x2)

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Accent)
plt.plot(x1, x2, c='red')
plt.show()    

# Accuracy
Y_ = getPredictions(X, weights, labels=True)
print(Y_, Y)    

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, Y_)
print(cm)

training_acc = np.sum(Y == Y_)/ Y.shape[0]
print(training_acc)