import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.tree import DecisionTreeClassifier, plot_tree
import utils
import B1.models

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  '''Load cartoon set image, crop to eye area
  Args:
    filename: string. The image filename to load
  Returns:
    numpy array (3d). The final image in RGB colour
  '''
  img = cv2.imread(filename) # this task needs colour
  crop = img[240:280,170:250,::-1] # crop to eye, reverse BGR to RGB
  return crop

def _enhance(X):
  '''Enhance the iris area by cropping and then using min max scaling
  Args:
    X: numpy array. The 4d image dataset to be enhanced
  Returns:
    numpy array. The 3d enhanced image dataset
  '''
  print('Peforming Iris Cropping and Contrast Enhancement...')
  X = X[:,25,20:35] # crop to a slice of the iris
  # Contrast stretching
  mn = X.min(axis=1,keepdims=True)
  mx = X.max(axis=1,keepdims=True)
  print(f'{(mn==mx).all(axis=-1).mean():.2%} of samples have no variance')
  return np.divide(X-mn,np.maximum((mx-mn),1), dtype=np.float32)*255 # Tree classifier creates copy if not float32

class IrisDTree:
  '''The decision tree class with iris enhacncement
  Attributes:
    model: sklearn.tree.DecisionTreeClassifier. The decision tree
  '''
  def fit(self, X, y):
    '''Fit the model with the given data
    Args:
      X: numpy array. The 4d(colour) image dataset
      y: numpy array. The labels
    Returns:
      float. The accuracy score on the dataset after fitting
    '''
    X = _enhance(X)
    X = utils.flatten(X)
    min_depth = int(max(y)).bit_length() # need at least this many nodes to distinguish all classes
    params = {'criterion':['gini', 'entropy'], 'max_depth':range(min_depth, min_depth+5)}
    self.model = DecisionTreeClassifier(random_state=RND, max_depth=min_depth)
    utils.cv_optimiser(self.model, X, y, params)
    B1.models.plot(self.model)
    return self.model.score(X,y)

  def predict(self, X):
    '''Run the model to predict the labels from a given dataset
    Args:
      X: numpy array. The 4d(colour) image dataset
    Returns:
      numpy array. The array of predicted labels
    '''
    X = _enhance(X) 
    X = utils.flatten(X)
    return self.model.predict(X)

# This dict of model options is used by the utils module to create a dropdown list
options = {'*Best B2: Decision Tree with CV paramater optimisation': B1.models.DTree(cv_optimise=True),
          'B2: Decision Tree with pre-optimsed parameters':B1.models.DTree(criterion='entropy', max_depth=9),
          'B2: Iris Enhance & Decision Tree with CV paramater optimisation': IrisDTree()}
