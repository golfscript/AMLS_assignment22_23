import numpy as np
import tensorflow as tf
import cv2

RND = 1 # use in all functions that need random seed in order to ensure repeatability
tf.random.set_seed(RND)

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
    crop = img[90:180, 54:124] # crop to face
    return crop/255.0

def fit(X, y, X_test=None, y_test=None):
    global model
    if X.ndim == 3:
        X = X.reshape(*X.shape,1) # reshape if necessary for Conv2D layer
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=X[0].shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(8, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2)
        ])
    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    
    model.fit(X, y, epochs=10, validation_data=(X_test, y_test))

    return model

def predict(X):
    global model
    if X.ndim == 3:
        X = X.reshape(*X.shape,1) # reshape if necessary for Conv2D layer
    return model.predict(X).argmax(axis=-1) # select prediction with highest output
