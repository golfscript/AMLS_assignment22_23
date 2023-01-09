import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import cv2
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv # need this to enable line below
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[90:180, 54:124] # crop to face
  return crop

def _augment(X, y):
  print('Performing Data Augmentation...')
  n = len(X)
  y = np.tile(y,2)
  X = np.tile(X,(2,1,1))
  X[n:] = X[:n,:,::-1]  # copy images, but reverse left to right

  plt.figure(figsize=(12,3))
  for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.title(f'[{i}]')
    plt.imshow(X[i+n], cmap='gray')
  plt.show()

  return X, y

def _clahe(X):
  eq = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,6)) # choose grid that divides 70*90 image
  Z = X.copy()
  for i in range(len(Z)):
    Z[i] = eq.apply(Z[i])
  return Z

def _prepare(X, y, clahe, augment):
  if clahe: X = _clahe(X)
  if augment: X, y = _augment(X, y)
  return X,y

def flat_scale(X):
  return np.divide(X.reshape((len(X),-1)),255.,dtype=np.float32)

class base:
  def __init__(self, pca_components=60, clahe=True, augment=True):
    self.model = make_pipeline(FunctionTransformer(flat_scale), PCA(n_components=pca_components,random_state=RND), StandardScaler(), SVC())
    self.clahe = clahe
    self.augment = augment

  def predict(self, X):
    X,_ = _prepare(X, None, self.clahe, False) # don't need to augment prediction
    return self.model.predict(X)

class PCA_SVC_Optimise(base):
  def fit(self, X, y):
    print('Peforming Cross Validation on Contrast Limited Adaptive Histogram Equalisation...')
    best_score = cross_val_score(self.model, X, y, n_jobs=-1).mean()

    Xc = _clahe(X)
    clahe_score = cross_val_score(self.model, Xc, y, n_jobs=-1).mean()
    plt.plot(['No CLAHE', 'CLAHE'], [best_score, clahe_score])
    plt.show()

    self.clahe = clahe_score > best_score # use CLAHE if cv score is better
    if self.clahe:
      X = Xc
      best_score = clahe_score

    params = {'svc__kernel':['linear','poly','rbf','sigmoid'],
              'svc__C':[0.1, 1, 10],
              'svc__gamma':[0.001, 'scale', 0.1],
              'pca__n_components':[80, 100, 120, 140]}
    
    for param, values in params.items():
      print(f'Peforming Cross Validation on optimal {param}...')
      prog_bar = tqdm(values, desc='cross validation')
      scores = [cross_val_score(self.model.set_params(**{param:v}), X, y, n_jobs=-1).mean() for v in prog_bar]
      plt.plot([str(v) for v in values], scores)
      plt.show()

      best = values[np.argmax(scores)]
      best_score = max(scores)
      print(f'Optimal {param} is', best)
      self.model.set_params(**{param:best})
    
    X, y = _augment(X, y)
  
    print('Peforming Cross Validation on Contrast Limited Adaptive Histogram Equalisation...')
    aug_score = cross_val_score(self.model, X, y, n_jobs=-1).mean()
    plt.plot(['No augmentation', 'Data augmentaion'], [best_score, aug_score])
    plt.show()

    self.augment = aug_score>best_score
    print('Performing final fit on all data with optimal params...')
    if (self.augment):
      self.model.fit(X,y)
    else:
      self.model.fit(X[:len(X)//2],y[:len(y)//2])

    return self.model.score(X,y)

class PCA_SVC_HCV(base):
  def fit(self, X, y):
    X,y = _prepare(X, y, self.clahe, self.augment)
    
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
    X, y = _prepare(X, y, self.clahe, self.augment)

    print('Performing Support Vector Classification Fit...')
    self.model.fit(X,y)
    return self.model.score(X,y)

options = {'*Best A1: PCA & SVC with CV optimised paramaters': PCA_SVC_Optimise(),
          'Clahe, Augment, PCA & SVC Halving Grid Search with Cross Validation': PCA_SVC_HCV(120),
          'Clahe, Augment, PCA(120) & SVC(default params)': PCA_SVC(120)}
