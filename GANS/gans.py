import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l1_l2 as l1l2
from keras.datasets import mnist
from keras.optimizers import Adam

np.random.seed(100)
randomD = 100

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train/255.0
X_test = X_test/255.0

X_train = X_train.reshape(60000, 784)

adam = Adam(lr=0.0002, beta_1=0.5)

generator = Sequential()
generator.add(Dense(256, input_dim=randomD, activation='relu'))
generator.add(Dense(512, activation='relu'))
generator.add(Dense(1024, activation='relu'))
generator.add(Dense(784, activation='sigmoid'))

generator.compile(loss='binary_crossentropy', optimizer=adam)
generator.summary()


discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784, activation='relu'))
discriminator.add(Dense(512, activation='relu'))
discriminator.add(Dense(256, activation='relu'))
discriminator.add(Dense(64, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=adam)
discriminator.summary()


discriminator.trainable=False
ganInp = Input(shape=(randomD, ))
x = generator(ganInp)
ganOut = discriminator(x)
gan = Model(input=ganInp, output=ganOut)

gan.compile(loss='binary_crossentropy', optimizer='adam')


dLosses = []
gLosses = []

def plot_loss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminator Loss')
    plt.plot(gLosses, lable='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('gan_loss_epoch_%d.png' %epoch)
    
def plotGeneratedImages(epoch, example=20, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomD])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)
    
    plt.figure(figsize=figsize)
    for ix in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], ix+1)
        plt.imshow(generatedImages[ix], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gen_generated_image_epoch_%d.png' %epoch)

def save_models(epoch):
    generator.save('models/gan_gen_epoch_%d.h5' %epoch)
    discriminator.save('model/gan_dis_epoch_%d.h5)
    
def train(epochs=1, batchsize=128):
    batchCount = X_train.shape[0] / batchsize
    
    print("Epochs: ", epochs)
    print("Batch Size: ", batchsize)
    print("Batches per epoch: ", batchCount)
    plotGeneratedImages(0)
    for e in range(1, epochs+1):
        print('_'*15, 'Epoch %d', %e, '_'*15)
        for _ in tqdm(range(batchCount)):
            noise = np.random.normal(0, 1, size=[batchsize, randomD])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], )]
            generatedIamges = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImage])
            yDis = np.zeros(2*batchsize)
            yDis = [:Batchsize] = 0.9
            
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)
            
            noise = np.random.normal(0, 1, size = [batchsize, randomD])
            yGen = np.ones(batchsize)
            discriminator.trainable = False
            gloss = gan.train_on_batches(noise, yGen)
            
        dLosses.append(dloss)
        gLosses.append(gloss)
        
        if e == 1 or e%10==0:
            plotGeneratedImages(e)
            save_model(e)
    
    plot_loss(e)









