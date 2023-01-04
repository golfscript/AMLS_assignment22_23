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
  return exposure.equalize_adapthist(crop/255.0)*255 # apply adaptive histogram equalisation

def _prepare(X):
  X = X.reshape(len(X),-1) # flatten the last dims of X without copying
  return np.divide(X,255,dtype=np.float32) # convert to [0,1] and limit to float32 to save space

def _plot(pca):
  plt.title(f'Cumulative sum of variance explanied by #components {pca.explained_variance_ratio_.sum():.0%}')
  plt.plot(pca.explained_variance_ratio_.cumsum())
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.show()

class PCA_SVC_HCV:
  def __init__(self, pca_components=120):
    self.pca = PCA(n_components=pca_components, random_state=RND)
    self.svc = SVC()

  def fit(self, X, y):
    X = _prepare(X)
    print('Performing Principle Component Analysis')
    X = self.pca.fit_transform(X) # do fit & transform in one step as quicker
    _plot(self.pca)

    print('Performing Halving Grid Search with Cross Validation')
    Cs = [0.5, 1, 2, 5, 10]
    param_grid = [{'kernel':['linear','poly'], 'C':Cs},
                  {'kernel': ['rbf','sigmoid'], 'C':Cs, 'gamma':[0.001, 0.003, 0.01, 0.03, 0.1]}]
    self.model = HalvingGridSearchCV(self.svc, param_grid, max_resources=len(X)//2, random_state=RND, verbose=3).fit(X, y)
    print("Optimal hyper-parameters:", self.model.best_params_)
    print("Mean cross-validated accuracy:", self.model.best_score_)
    return self.model.score(X,y)

  def predict(self, X):
    X = _prepare(X)
    X = self.pca.transform(X)
    return self.model.predict(X)

class PCA_SVC:
  def __init__(self, pca_components):
    self.pca = PCA(n_components=pca_components, random_state=RND)
    self.svc = SVC()

  def fit(self, X, y):
    X = _prepare(X)
    print('Performing Principle Component Analysis')
    X = self.pca.fit_transform(X) # do fit & transform in one step as quicker
    _plot(self.pca)

    print('Performing Support Vector Classification')
    self.svc.fit(X,y)
    return self.svc.score(X,y)
  
  def predict(self, X):
    X = _prepare(X)
    X = self.pca.transform(X)
    return self.svc.predict(X)

models = {'Best: PCA(120) & SVC Halving Grid Search with Cross Validation': PCA_SVC_HCV(120),
          'PCA(120) & SVC with default params': PCA_SVC(120),
          'PCA(100) & SVC with default params': PCA_SVC(100),
          'PCA(140) & SVC with deafult params': PCA_SVC(140)}