# Part - 1 Data Preprocessing


# Importing libraries and boston dataset
import numpy as np 
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Getting features and targets
boston = load_boston()
features = np.array(boston.data)
labels = np.array(boston.target)
labels = np.reshape(labels, (-1, 1))

# Normalising Function
def normalise(data):
    m = np.mean(data)
    s = np.std(data)
    normalised_data = (data - m)/s
    return normalised_data

# Applying normalising function to features and labels
features = normalise(features)
labels = normalise(labels)

# Appending x0 = 1 to features
bias_feature = np.ones(shape=(features.shape[0], 1))
features = np.concatenate((bias_feature, features), axis=1)

# Splitting dataset
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)


# Part - 2 Creating Model


# Assigning variables
n_samples = features.shape[0]
n_features = features.shape[1]
num_epochs = 40

# Creating Placeholders for inputs
X = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Creating weight matrix
W = tf.Variable(tf.random_normal(shape=(n_features, 1)))

# Predictions
y_pred = tf.matmul(X, W)

# Cost
cost = tf.reduce_mean(tf.square(y_pred - Y))

# Optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# Part - 3 Execute Graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        sess.run(opt, feed_dict = {
                X: X_train,
                Y: Y_train
                })
        
        train_loss = sess.run(cost, feed_dict = {
                        X: X_train,
                        Y: Y_train
                        })
        
        test_loss = sess.run(cost, feed_dict = {
                    X: X_test,
                    Y: Y_test
                    })
        
        print('Epoch{}\n'.format(epoch+1))
        print('Training Loss is {:.04f} and Testing Loss is {:.04f}'.format(train_loss, test_loss))
        print('---------------------')



