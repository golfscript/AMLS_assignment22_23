import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename) # this task needs colour
  crop = img[240:280,160:340,::-1] # crop to eyes, reverse GBR to RGB
  return crop

def _prepare(X):
  return X.reshape(len(X),-1) # flatten the last dims of X without copying

def _plot(model):
  plt.figure(figsize=(12,10))
  plot_tree(model, fontsize=9) # plot the decision tree
  plt.show()

def _crop_enhance(X):
  print('Peforming Iris Cropping and Contrast Enhancement...')
  X = X[:,25,30:45] # crop to iris
  # Contrast stretching
  mn = X.min(axis=1,keepdims=True)
  mx = X.max(axis=1,keepdims=True)
  print(f'{(mn==mx).all(axis=-1).mean():.2%} of samples have no variance')
  return np.divide(X-mn,np.maximum((mx-mn),1), dtype=np.float32)*255 # Tree classifier creates copy if not float32

class DTreeCV:
  def __init__(self, crop_enhance):
    self.crop_enhance = crop_enhance
  
  def fit(self, X, y):
    if self.crop_enhance: X = _crop_enhance(X)
    X = _prepare(X)
    min_depth = y.max() # need at least this many nodes to distinguish all classes
    params = {'criterion':['gini', 'entropy'], 'max_depth':range(min_depth, min_depth+6)}
    self.model = GridSearchCV(DecisionTreeClassifier(random_state=RND), params, verbose=3).fit(X,y)
    print("Optimal hyper-parameters:", self.model.best_params_)
    print("Mean cross-validated accuracy:", self.model.best_score_)
    _plot(self.model.best_estimator_)
    return self.model.score(X,y)

  def predict(self, X):
    if self.crop_enhance: X = _crop_enhance(X) 
    X = _prepare(X)
    return self.model.predict(X)

options = {'*Best B2: Iris Crop/Enhance & Decision Tree with CV': DTreeCV(True),
          'Decision Tree with CV': DTreeCV(False)}
