import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import cv2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

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
  plot_tree(model, fontsize=9) # plot the decision tree
  plt.show()

class DecisionTree:
  def __init__(self):
    self.model = DecisionTreeClassifier(random_state=RND)

  def fit(self, X, y):
    X = _prepare(X)
    min_depth = y.max() # need at least this many nodes to distinguish all classes
    self.model.set_params(max_depth=min_depth)
    params = {'criterion':['gini', 'entropy'], 'max_depth':range(min_depth, min_depth+4)}
    
    for param, values in params.items():
      print(f'Peforming Cross Validation on optimal {param}...')
      prog_bar = tqdm(values, desc='cross validation')
      scores = [cross_val_score(self.model.set_params(**{param:v}), X, y, n_jobs=-1).mean() for v in prog_bar]
      plt.plot([str(v) for v in values], scores)
      plt.show()

      best = values[np.argmax(scores)]
      print(f'Optimal {param} is', best)
      self.model.set_params(**{param:best})

    print('Performing final fit on all data with optimal params...')
    self.model.fit(X, y)
    _plot(self.model)
    return self.model.score(X,y)

  def predict(self, X):
    X = _prepare(X)
    return self.model.predict(X)

options = {'*Best B1: Decision Tree with CV of optimal params': DecisionTree()}