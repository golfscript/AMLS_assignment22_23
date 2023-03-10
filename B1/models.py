import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import cv2
from sklearn.tree import DecisionTreeClassifier, plot_tree
import utils

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  '''Load image, converting to greyscale and centre cropping to 200x200, then downscaling to 100x100
  Args:
    filename: string. The image filename to load
  Returns:
    numpy array (2d). The final image
  '''
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[200:400:2,150:350:2] # crop to face, skip every other pixel
  return crop

def plot(model):
  '''Display the given decision tree model
  Args:
    model: sklearn.tree.DecisionTreeClassifier. The decision tree
  '''
  plt.figure(figsize=(12,10))
  plot_tree(model, fontsize=9) # plot the decision tree
  plt.show()

class DTree:
  '''Decision tree model class with optional cross-validated optimisation of parameters
  Attributes:
    model: sklearn.tree.DecisionTreeClassifier. The decision tree
    cv_optimise: boolean. Whether to use cross-validation to optimise the parameters
  '''
  def __init__(self, cv_optimise=False, **kwargs):
    self.model = DecisionTreeClassifier(random_state=RND, **kwargs)
    self.cv_optimise = cv_optimise

  def fit(self, X, y):
    '''Fit the model with the given data
    Args:
      X: numpy array. The 3d(grayscale) or 4d(colour) image dataset
      y: numpy array. The labels
    Returns:
      float. The accuracy score on the dataset after fitting
    '''
    X = utils.flatten(X)
    if self.cv_optimise:
      min_depth = int(max(y)).bit_length() # need at least this many nodes to distinguish all classes
      self.model.set_params(max_depth=min_depth)
      params = {'criterion':['gini', 'entropy'], 'max_depth':range(min_depth, min_depth+10,3)}
      utils.cv_optimiser(self.model, X, y, params)
    else:
      print('Peforming fit on all data with selected params...')
      self.model.fit(X, y)
    
    plot(self.model)
    return self.model.score(X,y)

  def predict(self, X):
    '''Run the model to predict the labels from a given dataset
    Args:
      X: numpy array. The 3d(grayscale) or 4d(colour) image dataset
    Returns:
      numpy array. The array of predicted labels
    '''
    X = utils.flatten(X)
    return self.model.predict(X)

# This dict of model options is used by the utils module to create a dropdown list
options = {'*Best B1: Decision Tree with CV paramater optimisation': DTree(cv_optimise=True),
           'B1: Decision Tree with pre-optimsed parameters': DTree(criterion='entropy', max_depth=6)}