import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import cv2
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv # need this to enable line below
from sklearn.model_selection import HalvingGridSearchCV

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[90:180, 54:124] # crop to face
  return crop

def _clahe(X): # Contrast limited adaptive histogram equalisation
  print("Performing Contrast Limited Adaptive Histogram Equalisation...")
  for i in range(len(X)):
    X[i] = exposure.equalize_adapthist(X[i])

def _prepare(X, clahe):
  X = np.divide(X,255,dtype=np.float32) # convert to [0,1] and limit to float32 to save space
  if clahe: _clahe(X)
  return X.reshape(len(X),-1) # flatten the last dims of X without copying

def _plot(pca):
  plt.title(f'Cumulative sum of variance explanied by {pca.n_components_} componets = {pca.explained_variance_ratio_.sum():.0%}')
  plt.plot(pca.explained_variance_ratio_.cumsum())
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.show()

class PCA_SVC_HCV:
  def __init__(self, pca_components=120, clahe=True):
    self.pca = PCA(n_components=pca_components, random_state=RND)
    self.svc = SVC()
    self.clahe = clahe

  def fit(self, X, y):
    X = _prepare(X, self.clahe)
    print('Performing Principle Component Analysis...')
    X = self.pca.fit_transform(X) # do fit & transform in one step as quicker
    _plot(self.pca)

    print('Performing Halving Grid Search with Cross Validation...')
    Cs = [0.5, 1, 2, 5, 10]
    param_grid = [{'kernel':['linear','poly'], 'C':Cs},
                  {'kernel': ['rbf','sigmoid'], 'C':Cs, 'gamma':[0.001, 0.003, 0.01, 0.03, 0.1]}]
    self.model = HalvingGridSearchCV(self.svc, param_grid, max_resources=len(X)//2, random_state=RND, verbose=3).fit(X, y)
    print("Optimal hyper-parameters:", self.model.best_params_)
    print("Mean cross-validated accuracy:", self.model.best_score_)
    return self.model.score(X,y)

  def predict(self, X):
    X = _prepare(X, self.clahe)
    X = self.pca.transform(X)
    return self.model.predict(X)

class PCA_SVC:
  def __init__(self, pca_components, clahe=True):
    self.pca = PCA(n_components=pca_components, random_state=RND)
    self.svc = SVC()
    self.clahe = clahe

  def fit(self, X, y):
    X = _prepare(X, self.clahe)
    print('Performing Principle Component Analysis...')
    X = self.pca.fit_transform(X) # do fit & transform in one step as quicker
    _plot(self.pca)

    print('Performing Support Vector Classification Fit...')
    self.svc.fit(X,y)
    return self.svc.score(X,y)
  
  def predict(self, X):
    X = _prepare(X, self.clahe)
    X = self.pca.transform(X)
    return self.svc.predict(X)

options = {'Best A1: Clahe, PCA(120) & SVC Halving Grid Search with Cross Validation': PCA_SVC_HCV(120),
          'Clahe, PCA(120) & SVC(default params)': PCA_SVC(120),
          'Clahe, PCA(100) & SVC with default params': PCA_SVC(100),
          'Clahe, PCA(140) & SVC with deafult params': PCA_SVC(140),
          'PCA(120) & SVC Halving Grid Search with Cross Validation': PCA_SVC_HCV(120, clahe=False),
          'PCA(120) & SVC with default params': PCA_SVC(120, clahe=False)}