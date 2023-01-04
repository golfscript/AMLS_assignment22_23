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
  def __init__(self, cnn_layer1, cnn_layer2, activation='relu', epochs=5):
    self.cnn_layer1 = cnn_layer1
    self.cnn_layer2 = cnn_layer2
    self.activation = activation
    self.epochs = epochs

  def fit(self, X, y, X_test=None, y_test=None):
    X = _prepare(X)
    tf.keras.utils.set_random_seed(RND)
    classes = y.max()+1 # generalise to work with any number of classed
    self.model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255), # rescale
      tf.keras.layers.Conv2D(self.cnn_layer1, 3, activation=self.activation),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(self.cnn_layer2, 3, activation=self.activation),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(classes)
      ])
    self.model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    
    return self.model.fit(X, y, epochs=self.epochs, validation_data=(X_test, y_test)).history['accuracy'][-1]

  def predict(self, X):
    X = _prepare(X)
    return self.model.predict(X).argmax(axis=-1) # select prediction with highest output


models = {'Best: CNN Layer1: 4, CNN Layer2: 4':CNN(4,4),
          'CNN Layer1: 4, CNN Layer2: 4, sigmoid activation': CNN(4,4,'sigmoid')}