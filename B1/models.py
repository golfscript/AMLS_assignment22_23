import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn import tree

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[160:400,150:350] # crop to face
  scale = cv2.resize(crop,None,fx=1/2,fy=1/2,interpolation=cv2.INTER_NEAREST_EXACT) # rescale to one third size
  return scale

def _prepare(X):
  return X.reshape(len(X),-1) # flatten the last dims of X without copying

def _plot(model):
  plt.figure(figsize=(12,10))
  tree.plot_tree(model, fontsize=9) # plot the decision tree
  plt.show()

class DecisionTree:
  def __init__(self, max_depth):
    self.max_depth = max_depth

  def fit(self, X, y):
    X = _prepare(X)
    print('Peforming Decision Tree Fitting')
    self.model = tree.DecisionTreeClassifier(max_depth=self.max_depth, random_state=RND).fit(X, y)
    _plot(self.model)
    return self.model.score(X,y)

  def predict(self, X):
    X = _prepare(X)
    return self.model.predict(X) # select prediction with highest output

options = {'Best: Decision Tree with max depth of 6': DecisionTree(6),
          'Decision Tree with max depth of 5': DecisionTree(5),
          'Decision Tree with max depth of 4': DecisionTree(4),
          'Decision Tree with max depth of 3': DecisionTree(3)}