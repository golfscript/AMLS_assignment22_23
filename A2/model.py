import numpy as np
import tensorflow as tf
import cv2

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
    crop = img[90:180, 54:124] # crop to face
    return crop

def fit(X, y, X_test=None, y_test=None):
    global model
    if X.ndim == 3:
        X = X.reshape(*X.shape,1) # reshape if necessary for Conv2D layer
    tf.keras.utils.set_random_seed(RND)
    classes = y.max()+1 # generalise to work with any number of classed
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(4, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(classes)
        ])
    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    
    return model.fit(X, y, epochs=6, validation_data=(X_test, y_test)).history['accuracy'][-1]

def predict(X):
    global model
    if X.ndim == 3:
        X = X.reshape(*X.shape,1) # reshape if necessary for Conv2D layer
    return model.predict(X).argmax(axis=-1) # select prediction with highest output
