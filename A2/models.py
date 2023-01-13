from matplotlib import pyplot as plt
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # suppress TensorFlow info messages
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

RND = 1 # use in all functions that need random seed in order to (try to) ensure repeatability

def load_image(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[90:180, 54:124] # crop to face
  return crop

def _prepare(X):
  return X.reshape(*X.shape,1) if X.ndim == 3 else X # reshape if necessary for Conv2D layer

class CNN:
  def __init__(self, cnn_layers=(), kernel_size=3, pool_size=2, dense_layers=(), activation='relu', dropout=0.3, regularizer='l2', epochs=10, weights_file=None):
    self.cnn_layers = cnn_layers
    self.pool_size = pool_size
    self.kernel_size = kernel_size
    self.dense_layers = dense_layers
    self.activation = activation
    self.dropout = dropout
    self.regularizer = regularizer
    self.epochs = epochs
    self.weights_file = weights_file

  def fit(self, X, y):
    X = _prepare(X)
    tf.keras.utils.set_random_seed(RND)
    self.model = tf.keras.Sequential([tf.keras.layers.Rescaling(1./255, input_shape=X.shape[1:]), # rescale
      tf.keras.layers.RandomFlip(mode='horizontal')]) # data augmentation

    for n in self.cnn_layers: # add CNN layers
      self.model.add(tf.keras.layers.Conv2D(n, self.kernel_size, activation=self.activation))
      self.model.add(tf.keras.layers.MaxPooling2D(self.pool_size))
      if self.dropout>0:
        self.model.add(tf.keras.layers.SpatialDropout2D(self.dropout))
  
    self.model.add(tf.keras.layers.Flatten())
    for n in self.dense_layers: # add dense layers
      self.model.add(tf.keras.layers.Dense(n, activation=self.activation, kernel_regularizer=self.regularizer))
      if self.dropout>0:
        self.model.add(tf.keras.layers.Dropout(self.dropout))

    final_layer = y.max()+1
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy
    if final_layer == 2:
      final_layer = 1 # if only two classes then it's a binary problem
      loss_fn = tf.keras.losses.BinaryCrossentropy

    self.model.add(tf.keras.layers.Dense(final_layer, activation='sigmoid', kernel_regularizer=self.regularizer)) # final layer for classification

    self.model.summary() # print a summary of the model
    self.model.compile(optimizer='adam', loss=loss_fn(from_logits=False), metrics=['accuracy'])

    if self.weights_file: # if load file specified
      self.model.load_weights(self.weights_file) # then load weights
    
    if self.epochs==0: return self.model.evaluate(X, y)[1] # if epochs=0, then just return evaluation

    print('Performing validated assessment of model accuracy...')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RND)
    history = self.model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_val, y_val)).history

    plt.plot([a*100 for a in history['accuracy']])
    plt.plot([a*100 for a in history['val_accuracy']])
    plt.title('Accuracy after each epoch')
    plt.xlabel('epoch')
    plt.ylabel('% accuracy')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

    print('Performing final fit on all data...')
    history = self.model.fit(X, y, epochs=self.epochs).history

    return history['accuracy'][-1] # return latest accuracy score

  def predict(self, X):
    X = _prepare(X)
    y = self.model.predict(X)
    if y.shape[1]>1: return y.argmax(axis=-1) # select prediction with highest output
    return (y.reshape(-1)>=0.5)*1 # convert to 0 or 1

options = {'*Best A2: Small CNN': CNN((4,4), pool_size=3, epochs=60),
          'A2: Small CNN (saved weights)': CNN((4,4), pool_size=3, epochs=0, weights_file='A2/cnn44pool3'),
          'A2: Tiny CNN':CNN((2,2),pool_size=4,epochs=60),
          'A2: Medium CNN':CNN((8,16),dense_layers=(128,),epochs=30),
          'A2: Large CNN':CNN((32,64,128),dense_layers=(256,))}
  