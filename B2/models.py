import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.tree import DecisionTreeClassifier, plot_tree
import utils
import B1.models

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename) # this task needs colour
  crop = img[240:280,170:250,::-1] # crop to eye, reverse BGR to RGB
  return crop

def _prepare(X):
  return X.reshape(len(X),-1) # flatten the last dims of X without copying

def _plot(model):
  plt.figure(figsize=(12,10))
  plot_tree(model, fontsize=9) # plot the decision tree
  plt.show()

def _crop_enhance(X):
  print('Peforming Iris Cropping and Contrast Enhancement...')
  X = X[:,25,20:35] # crop to iris
  # Contrast stretching
  mn = X.min(axis=1,keepdims=True)
  mx = X.max(axis=1,keepdims=True)
  print(f'{(mn==mx).all(axis=-1).mean():.2%} of samples have no variance')
  return np.divide(X-mn,np.maximum((mx-mn),1), dtype=np.float32)*255 # Tree classifier creates copy if not float32

class IrisDTree:
  def fit(self, X, y):
    X = _crop_enhance(X)
    X = _prepare(X)
    min_depth = int(max(y)).bit_length() # need at least this many nodes to distinguish all classes
    params = {'criterion':['gini', 'entropy'], 'max_depth':range(min_depth, min_depth+5)}
    self.model = DecisionTreeClassifier(random_state=RND, max_depth=min_depth)
    utils.cv_optimiser(self.model, X, y, params)
    _plot(self.model)
    return self.model.score(X,y)

  def predict(self, X):
    X = _crop_enhance(X) 
    X = _prepare(X)
    return self.model.predict(X)

options = {'*Best B2: Decision Tree with CV paramater optimisation': B1.models.DTree(cv_optimise=True),
          'B2: Decision Tree with pre-optimsed parameters':B1.models.DTree(criterion='entropy', max_depth=9),
          'B2: Iris Enhance & Decision Tree with CV paramater optimisation': IrisDTree()}
