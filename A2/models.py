import numpy as np
import tensorflow as tf
import cv2

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[90:180, 54:124] # crop to face
  return crop

def _prepare(X):
  return X.reshape(*X.shape,1) if X.ndim == 3 else X # reshape if necessary for Conv2D layer

class CNN:
  def __init__(self, cnn_layers=(), pool_size=2, dense_layers=(), activation='relu', dropout=0, regularizer=None, epochs=6, weights_file=None):
    self.cnn_layers = cnn_layers
    self.pool_size = pool_size
    self.dense_layers = dense_layers
    self.activation = activation
    self.dropout = dropout
    self.regularizer = regularizer
    self.epochs = epochs
    self.weights_file = weights_file

  def fit(self, X, y, X_test=None, y_test=None):
    X = _prepare(X)
    tf.keras.utils.set_random_seed(RND)
    self.model = tf.keras.Sequential([tf.keras.layers.Rescaling(1./255, input_shape=X.shape[1:])]) # rescale

    for n in self.cnn_layers: # add CNN layers
      self.model.add(tf.keras.layers.Conv2D(n, 3, activation=self.activation))
      self.model.add(tf.keras.layers.MaxPooling2D(self.pool_size))
      if self.dropout>0:
        self.model.add(tf.keras.layers.SpatialDropout2D(self.dropout))
  
    self.model.add(tf.keras.layers.Flatten())
    for n in self.dense_layers: # add dense layers
      self.model.add(tf.keras.layers.Dense(n, activation=self.activation, kernel_regularizer=self.regularizer))
      if self.dropout>0:
        self.model.add(tf.keras.layers.Dropout(self.dropout))

    self.model.add(tf.keras.layers.Dense(y.max()+1)) # final layer for classification

    self.model.summary()

    if self.weights_file:
      self.model.load_weights(self.weights_file)

    self.model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    
    if self.epochs==0: return self.model.evaluate(X, y)[1]
    
    return self.model.fit(X, y, epochs=self.epochs, validation_data=(X_test, y_test)).history['accuracy'][-1]

  def predict(self, X):
    X = _prepare(X)
    return self.model.predict(X).argmax(axis=-1) # select prediction with highest output

options = {'*Best A2: CNN(4,4) pool size 3 relu': CNN((4,4), pool_size=3, epochs=30),
          'CNN(4,4) relu':CNN((4,4)),
          'CNN(4,4) sigmoid': CNN((4,4),activation='sigmoid'),
          'CNN(32,64,128) Dense(256) with dropout 0.3 & l2 reg, relu':CNN((32,64,128),dense_layers=(256,),dropout=0.3,regularizer='l2',epochs=10),
          'CNN(4,4) pool size 3, relu (saved weights)': CNN((4,4), pool_size=3, epochs=0, weights_file='A2/p622a9090')}