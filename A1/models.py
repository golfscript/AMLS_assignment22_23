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
  print("Performing Contrast Limited Adaptive Histogram Equalisation", end='')
  for i in range(len(X)):
    X[i] = exposure.equalize_adapthist(X[i])
    if i%100 == 0: print('.', end='')
  print()

def _prepare(X, clahe, augment):
  Z = np.empty((2 if augment else 1,*X.shape), dtype=np.float32) # limit to float32 to save space
  Z[0] = X/255. # convert to [0,1]
  if clahe: _clahe(Z[0])
  if augment:
    print('Performing Data Augmentation...')
    Z[1] = Z[0,:,:,::-1]  # copy images, but reverse left to right
  
  plt.figure(figsize=(12,3))
  for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.title(f'[{i}]')
    plt.imshow(Z[-1,i], cmap='gray')
  plt.show()

  return Z.reshape(len(Z)*len(X),-1) # flatten the last dims of Z without copying

def _plot(pca):
  plt.title(f'Cumulative sum of variance explanied by {pca.n_components_} componets = {pca.explained_variance_ratio_.sum():.0%}')
  plt.plot(pca.explained_variance_ratio_.cumsum())
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.show()

class base:
  def __init__(self, pca_components=120, clahe=True, augment=True):
    self.pca = PCA(n_components=pca_components, random_state=RND)
    self.clahe = clahe
    self.augment = augment

  def predict(self, X):
    X = _prepare(X, self.clahe, False)
    X = self.pca.transform(X)
    return self.model.predict(X)

class PCA_SVC_HCV(base):
  def fit(self, X, y):
    X = _prepare(X, self.clahe, self.augment)
    if self.augment: y = np.tile(y,2) # duplicate labels

    print('Performing Principle Component Analysis...')
    X = self.pca.fit_transform(X) # do fit & transform in one step as quicker
    _plot(self.pca)

    print('Performing Halving Grid Search with Cross Validation...')
    Cs = [0.5, 1, 2, 5, 10, 20]
    param_grid = [{'kernel':['linear','poly'], 'C':Cs},
                  {'kernel': ['rbf','sigmoid'], 'C':Cs, 'gamma':[0.001, 0.003, 0.01, 0.03, 0.1]}]
    self.model = HalvingGridSearchCV(SVC(), param_grid, max_resources=len(X)//2, random_state=RND, verbose=3).fit(X, y)
    print("Optimal hyper-parameters:", self.model.best_params_)
    print("Mean cross-validated accuracy:", self.model.best_score_)
    return self.model.score(X,y)

class PCA_SVC(base):
  def fit(self, X, y):
    X = _prepare(X, self.clahe, self.augment)
    if self.augment: y = np.tile(y,2) # duplicate labels

    print('Performing Principle Component Analysis...')
    X = self.pca.fit_transform(X) # do fit & transform in one step as quicker
    _plot(self.pca)

    print('Performing Support Vector Classification Fit...')
    self.model = SVC().fit(X,y)
    return self.model.score(X,y)

options = {'*Best A1: Clahe, Augment, PCA(120) & SVC Halving Grid Search with Cross Validation': PCA_SVC_HCV(120),
          'Clahe, Augment, PCA(120) & SVC(default params)': PCA_SVC(120),
          'Clahe, Augment, PCA(100) & SVC with default params': PCA_SVC(100),
          'Clahe, Augment, PCA(140) & SVC with deafult params': PCA_SVC(140),
          'Augment, PCA(120) & SVC Halving Grid Search with Cross Validation': PCA_SVC_HCV(120, clahe=False),
          'Augment, PCA(120) & SVC with default params': PCA_SVC(120, clahe=False)}