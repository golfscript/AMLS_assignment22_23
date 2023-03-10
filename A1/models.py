import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv # need this to enable line below
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import utils

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
  '''Load image, converting to greyscale and centre cropping to 70x90
  Args:
    filename: string. The image filename to load
  Returns:
    numpy array (2d). The final image
  '''
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
  crop = img[90:180, 54:124] # crop to face
  return crop

def _augment(X, y):
  '''Augment the given dataset by flipping each image horizontally
      X: numpy array. The 3d(grayscale) or 4d(colour) image dataset
      y: numpy array. The labels
    Returns:
      tuple of numpy arrays. The augmented image dataset, and augmented labels
    '''
  print('Performing Data Augmentation...')
  n = len(X)
  y = np.tile(y,2)
  X = np.tile(X,(2,1,1))
  X[n:] = X[:n,:,::-1]  # copy images, but reverse left to right

  utils.show_images(X[n:],y[n:])
  return X, y

def _clahe(X):
  '''Peform contrast limited adaptive histogram equalisation on an image dataset
  Args:
    X: numpy array. The 3d array of greyscale images
  Returns:
    numpy array. The 3d array of enhanced images
  '''
  print('Performing CLAHE...')
  eq = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,6)) # choose grid that excatly divides 70*90 image
  Z = X.copy()
  for i in range(len(Z)):
    Z[i] = eq.apply(Z[i]) # apply clahe to each image
  utils.show_images(Z)
  return Z

def _prepare(X, y, clahe, augment):
  '''Prepare the given dataset with optional CLAHE and augmentation
  Args:
    X: numpy array. The 3d(grayscale) or 4d(colour) image dataset
    y: numpy array. The labels
    Returns:
      tuple of numpy arrays. The prepared image dataset, and labels
    '''
  if clahe: X = _clahe(X)
  if augment: X, y = _augment(X, y)
  return X,y

def flat_scale(X):
  '''Flatten the given dataset, and scale from 0..255 to 0..1
  Args:
   X: numpy array. The image dataset
  Returns:
    numpy array. The flattened and scaled dataset
  '''
  X = utils.flatten(X)
  return np.divide(X,255.,dtype=np.float32)

class PCA_SVC:
  '''Base class for PCA and SVC model
  Attributes:
    model: sklearn.pipeline.Pipeline. The PCA and SVC model pipeline
    clahe: boolean. Whether to apply CLAHE before the pipeline
    augment: boolean. Whether to apply data augmentation before the pipline
  Methods:
    fit(X, y): fit the model to the given dataset and labels
    predict(X): predict labels from the given image dataset
  '''
  def __init__(self, pca_components=60, clahe=True, augment=True):
    self.model = make_pipeline(FunctionTransformer(flat_scale), PCA(n_components=pca_components,random_state=RND), StandardScaler(), SVC())
    self.clahe = clahe
    self.augment = augment

  def fit(self, X, y):
    '''Fit the model with the given data
    Args:
      X: numpy array. The 3d(grayscale) or 4d(colour) image dataset
      y: numpy array. The labels
    Returns:
      float. The accuracy score on the dataset after fitting
    '''
    X, y = _prepare(X, y, self.clahe, self.augment)

    print('Performing Support Vector Classification Fit...')
    self.model.fit(X,y)
    return self.model.score(X,y)

  def predict(self, X):
    '''Run the model to predict the labels from a given dataset
    Args:
      X: numpy array. The 3d(grayscale) or 4d(colour) image dataset
    Returns:
      numpy array. The array of predicted labels
    '''
    X,_ = _prepare(X, None, self.clahe, False) # don't need to augment prediction
    return self.model.predict(X)


class PCA_SVC_Optimise(PCA_SVC):
  '''PCA and SVC model class with cross-validated optimisation of all paramaters'''
  def fit(self, X, y):
    '''Train the model with the given data
    Args:
      X: numpy array. The 3d(grayscale) or 4d(colour) image dataset
      y: numpy array. The labels
    Returns:
      float. The accuracy score on the dataset after training
    '''
    if X.ndim!=3: # If colour images, don't try CLAHE
      self.clahe = False
    else: # If greyscale images, try CLAHE
      print('Peforming Cross Validation on Contrast Limited Adaptive Histogram Equalisation...')
      best_score = cross_val_score(self.model, X, y).mean()

      Xc = _clahe(X)
      clahe_score = cross_val_score(self.model, Xc, y).mean()
      utils.cv_plot('CLAHE', ['Raw images', 'CLAHE'], [best_score, clahe_score])

      self.clahe = clahe_score > best_score # use CLAHE if cv score is better
      if self.clahe:
        X = Xc

    params = {'svc__kernel':['linear','poly','rbf','sigmoid'],
              'svc__C':[0.1, 1, 10],
              'svc__gamma':[0.001, 'scale', 0.1],
              'pca__n_components':[60, 80, 100, 120, 140]}

    best_score = utils.cv_optimiser(self.model, X, y, params, refit=False)
    
    print('Peforming Cross Validation on Data Augmentation...')
    X, y = _augment(X, y)
    aug_score = cross_val_score(self.model, X, y).mean()
    utils.cv_plot('data augmentation', ['No augmentation', 'Data augmentaion'], [best_score, aug_score])

    self.augment = aug_score>best_score # use data augmentation if cv score is better
    print('Performing final fit on all data with optimal params...')
    if (self.augment):
      self.model.fit(X,y)
    else:
      self.model.fit(X[:len(X)//2],y[:len(y)//2])

    return self.model.score(X,y)

class PCA_SVC_HCV(PCA_SVC):
  '''PCA and SVC model class with successive cv halving optimisation of some parameters'''
  def fit(self, X, y):
    '''Run the model to predict the labels from a given dataset
    Args:
      X: numpy array. The 3d(grayscale) or 4d(colour) image dataset
    Returns:
      numpy array. The array of predicted labels
    '''
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

# This dict of model options is used by the utils module to create a dropdown list
options = {'*Best A1: PCA & SVC with CV paramater optimisation': PCA_SVC_Optimise(),
          'A1: PCA & SVC with pre-optimised parameters': PCA_SVC(120),
          'A1: Halving Grid Search with Cross Validation': PCA_SVC_HCV(120)}
