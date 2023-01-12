import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import cv2
from sklearn.tree import DecisionTreeClassifier, plot_tree
import utils

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[200:400:2,150:350:2] # crop to face, skip every other pixel
  return crop

def _prepare(X):
  return X.reshape(len(X),-1) # flatten the last dims of X without copying

def _plot(model):
  plt.figure(figsize=(12,10))
  plot_tree(model, fontsize=9) # plot the decision tree
  plt.show()

class DTree:
  def __init__(self, cv_optimise=False, **kwargs):
    self.model = DecisionTreeClassifier(random_state=RND, **kwargs)
    self.cv_optimise = cv_optimise

  def fit(self, X, y):
    X = _prepare(X)
    if self.cv_optimise:
      min_depth = int(max(y)).bit_length() # need at least this many nodes to distinguish all classes
      self.model.set_params(max_depth=min_depth)
      params = {'criterion':['gini', 'entropy'], 'max_depth':range(min_depth, min_depth+10,3)}
      utils.cv_optimiser(self.model, X, y, params)
    else:
      print('Peforming fit on all data with selected params...')
      self.model.fit(X, y)
    
    _plot(self.model)
    return self.model.score(X,y)

  def predict(self, X):
    X = _prepare(X)
    return self.model.predict(X)

options = {'*Best B1: Decision Tree with CV of optimal params': DTree(cv_optimise=True),
           'Decision Tree with Entropy and max depth of 6': DTree(criterion='entropy', max_depth=6)}