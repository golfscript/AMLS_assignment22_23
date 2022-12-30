import numpy as np
import tensorflow as tf
from skimage import io, exposure

RND = 1 # use in all functions that need random seed in order to ensure repeatability
tf.random.set_seed(RND)

def load_image(filename):
    img = io.imread(filename, as_gray=True) # use built-in grayscale loading
    crop = img[90:180, 54:124] # crop to face
    return exposure.equalize_adapthist(crop)

def fit(X, y):
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
    
    model.fit(X, y, epochs=10)

    return model

def predict(X):
    global model
    if X.ndim == 3:
        X = X.reshape(*X.shape,1) # reshape if necessary for Conv2D layer
    return model.predict(X).argmax(axis=-1) # select prediction with highest output
