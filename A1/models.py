import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import cv2
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv # need this to enable line below
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[90:180, 54:124] # crop to face
  return crop

def _prepare(X, clahe, augment):
  Z = np.empty((2 if augment else 1,*X.shape), dtype=np.float32) # limit to float32 to save space
  if clahe:
    print("Performing Contrast Limited Adaptive Histogram Equalisation", end='')
    eq = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,6)) # choose grid that divides 70*90 image
    for i in range(len(X)):
      Z[0][i] = eq.apply(X[i])/255.
      if i%100 == 0: print('.', end='')
    print()
  else:
    Z[0] = X/255. # just convert to [0,1]

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
  def __init__(self, pca_components, clahe=True, augment=True):
    self.model = make_pipeline(PCA(n_components=pca_components,random_state=RND), StandardScaler(), SVC())
    self.clahe = clahe
    self.augment = augment

  def predict(self, X):
    X = _prepare(X, self.clahe, False)
    return self.model.predict(X)

class PCA_SVC_HCV(base):
  def fit(self, X, y):
    X = _prepare(X, self.clahe, self.augment)
    if self.augment: y = np.tile(y,2) # duplicate labels
    
    print('Performing Halving Grid Search with Cross Validation...')
    Cs = [0.01, 0.1, 1]
    param_grid = [{'pca__n_components':[100,120,140], 'svc__kernel':['linear','poly'], 'svc__C':Cs},
                  {'pca__n_components':[100,120,140], 'svc__kernel': ['rbf','sigmoid'], 'svc__C':Cs, 'svc__gamma':[0.001, 0.01, 0.1]}]
    hcv = HalvingGridSearchCV(self.model, param_grid, random_state=RND, verbose=3, refit=False).fit(X[:5000], y[:5000])
    print("Optimal hyper-parameters:", hcv.best_params_)
    print("Mean cross-validated accuracy:", hcv.best_score_)
    self.model.set_params(**hcv.best_params_) # set pipeline to best params
    self.model.fit(X,y) # fit to augmented data
    return self.model.score(X,y)

class PCA_SVC(base):
  def fit(self, X, y):
    X = _prepare(X, self.clahe, self.augment)
    if self.augment: y = np.tile(y,2) # duplicate labels

    print('Performing Support Vector Classification Fit...')
    self.model.fit(X,y)
    return self.model.score(X,y)

options = {'*Best A1: Clahe, Augment, PCA & SVC Halving Grid Search with Cross Validation': PCA_SVC_HCV(),
          'Clahe, Augment, PCA(120) & SVC(default params)': PCA_SVC(120),
          'Clahe, Augment, PCA(100) & SVC with default params': PCA_SVC(100),
          'Clahe, Augment, PCA(140) & SVC with deafult params': PCA_SVC(140),
          'Augment, PCA(120) & SVC Halving Grid Search with Cross Validation': PCA_SVC_HCV(120, clahe=False),
          'Augment, PCA(120) & SVC with default params': PCA_SVC(120, clahe=False),
          'PCA(120) & SVC with default params': PCA_SVC(120, clahe=False, augment=False),
          'Augment, PCA(140) & SVC with default params': PCA_SVC(140, clahe=False)}