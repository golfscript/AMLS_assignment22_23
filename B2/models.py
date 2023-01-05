import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn import tree

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename) # this task needs colour
  crop = img[240:280,160:340,::-1] # crop to eyes, reverse GBR to RGB
  return crop

def _prepare(X):
  return X.reshape(len(X),-1) # flatten the last dims of X without copying

def _plot(model):
  plt.figure(figsize=(12,10))
  tree.plot_tree(model, fontsize=9) # plot the decision tree
  plt.show()

def _crop_enhance(X):
  X = X[:,25,30:45].astype(np.uint16)
  # Contrast stretching
  mn = X.min(axis=1,keepdims=True)
  mx = X.max(axis=1,keepdims=True)
  return 255*(X-mn)//np.maximum((mx-mn),1)

class DecisionTree:
  def __init__(self, max_depth, crop_enhance):
    self.model = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=RND)
    self.crop_enhance = crop_enhance

  def fit(self, X, y):
    if self.crop_enhance: X = _crop_enhance(X) 
    X = _prepare(X)
    print('Peforming Decision Tree Fitting')
    self.model.fit(X, y)
    _plot(self.model)
    return self.model.score(X,y)

  def predict(self, X):
    if self.crop_enhance: X = _crop_enhance(X) 
    X = _prepare(X)
    return self.model.predict(X) # select prediction with highest output

options = {'*Best B2: Crop, Enhance & Decision Tree max depth of 4': DecisionTree(4, True),
          'Decision Tree max depth of 8': DecisionTree(8, False),
          'Decision Tree max depth of 7': DecisionTree(7, False),
          'Decision Tree max depth of 6': DecisionTree(6, False)}