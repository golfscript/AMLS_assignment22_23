'''Utiliity functions for loading data, displaying images & charts, and sequential optimisation of parameters'''
import numpy as np
from os import path
from sklearn.model_selection import cross_val_score
import A1.models as A1, A2.models as A2, B1.models as B1, B2.models as B2
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import ipywidgets as widgets
import importlib

# These constants concern the location and filenames of the datasets

LABELS = 'labels.csv' # name of the labels file
DATASETS = 'Datasets' # top level folder where all the datastets can be found
IMGS = 'img' # sub-folder for images
TEST = '_test' # folder suffix for test datasets

TASKS = {'A1':('celeba',2,1,A1), 'A2':('celeba',3,1,A2), 'B1':('cartoon_set',2,3,B1),'B2':('cartoon_set',1,3,B2)}
task_options = widgets.RadioButtons(options=TASKS, description='Task')

MODELS = {**A1.options, **A2.options, **B1.options, **B2.options}
model_options = widgets.Dropdown(options=MODELS, description='Model', layout=widgets.Layout(width='70%'))

def load_data(folder, feature_col, file_col, load_image, test=False):
  '''Loads a single dataset from the DATASETS folder
  Args:
    folder: string. The name of subfolder within DATASETS that contains the dataset.
    feature_col: integer. The column within the labels file that contains the feature of interest.
    file_col: integer. The column within the labels file that contains the filename of the image.
    load_image: function. The function for loading and pre-processing the image when passed the fully qualified filename.
      This function must return a numpy array, or array like object
    test: boolean. Whether to load the test version of the dataset.
      If this is true, then TEST is suffixed to the folder name.
    
  Returns:
    A tuple of numpy arrays (X, y)
      X is the main dataset of images
      y is the array of labels
 '''
  if test: folder += TEST # if loading test data add TEST to folder name
  filenames, y = np.genfromtxt(
      path.join(DATASETS,folder,LABELS),
      usecols=(file_col,feature_col),
      dtype='U10,i8',
      unpack=True,
      skip_header=1,
      delimiter='\t'
  )
  y = np.maximum(y,0) # convert -1 label to 0
  imagedir = path.join(DATASETS,folder,IMGS)
  n = len(filenames)
  shape = load_image(path.join(imagedir,filenames[0])).shape # get first image to get shape
  X = np.empty((n, *shape), dtype=np.uint8) # pre-define X, much more efficient than concatenating arrays
  for i in tqdm(range(n), desc=folder): # tqdm displays a nice loading bar
      X[i] = load_image(path.join(imagedir,filenames[i]))
  print(f'Loaded {X.nbytes:,} bytes')
  show_images(X[:5],y[:5])
  return X, y

def load_task_data():
  '''Loads the training and test datasets for a particular task, based on the value of the task_options radio button.
  Args:
    none
  Returns:
    A tuple of numpy arrays (X, y, X_test, y_test)
      X is the main dataset of images
      y is the array of labels
      X_test is the test dataset
      y_test is the array of test labels
  '''
  dataset, feature_col, file_col, models = task_options.value
  model_options.value = list(models.options.values())[0] # set default for model options
  X, y = load_data(dataset, feature_col, file_col, models.load_image)
  X_test, y_test = load_data(dataset, feature_col, file_col, models.load_image, test=True)
  return X, y, X_test, y_test

def show_images(X, y=None):
  '''Displays the first 5 images from a dataset, with optional labels
  Args:
    X: a 3d array of images
    y: optional array of labels
  Returns:
    none
  '''
  plt.figure(figsize=(12,3))
  for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.title(f'[{i}]' if y is None else f'[{i}] class:{y[i]}')
    plt.imshow(X[i], cmap='gray')
  plt.show()

def show_wrong(X, y, y_pred):
  '''Displays all the images from X images where y!=y_pred
  Args:
    X: a 3d array of images
    y: array of labels
    y_pred: array of predicted labels
  Returns:
    none
  '''
  wrong, = np.nonzero(y!=y_pred)
  print(f'There were {len(wrong)} wrong predictions out of {len(y)}')
  for w,i in enumerate(wrong):
    if w%5==0: plt.figure(figsize=(12,3))
    plt.subplot(1, 5, w%5+1)
    plt.axis('off')
    plt.title(f'[{i}] class:{y[i]} pred:{y_pred[i]}')
    plt.imshow(X[i], cmap='gray')
    if w%5==4: plt.show()

def cv_plot(name, values, scores):
  '''Plots cross valiation accuracy scores
  Args:
    name: string. The name of the parameter optimised by cross-validation
    values: a list or array of possible values for the parameter
    scores: a list or array of accuracy scores
  Returns:
    none
  '''
  plt.plot([str(v) for v in values], [s*100 for s in scores])
  plt.title(f'Mean cross-validated accuracy for {name}')
  plt.xlabel(name)
  plt.ylabel('% accuracy')
  plt.show()

def cv_optimiser(model, X, y, params, cv=5, refit=True):
  '''Sequential optimisation of model parameters using cross-validation
  Args:
    model: a Scikit-learn model
    X: numpy array. Training data
    y: numpy array. Labels
    params: dict. Parameter names:list of values
    cv: integer. Number of cross validation folds to use (default: 5)
    refit: boolean. Whether to fit the optimised model with all training data (default: True)
  Returns:
    best score in the last cross validation (or 0.0 if no validations)
  '''
  scores = [0.0] # in case there are no params to fit
  for param, values in params.items():
    name = param.replace('__', ' ')
    print(f'Peforming Cross Validation on optimal {name}...')
    prog_bar = tqdm(values, desc='cross validation')
    scores = [cross_val_score(model.set_params(**{param:v}), X, y, cv=cv).mean() for v in prog_bar]
    cv_plot(name, values, scores)
    best = values[np.argmax(scores)]
    print(f'Optimal {param} is', best)
    model.set_params(**{param:best})
  if refit:
    print('Performing final fit on all data with optimal params...')
    model.fit(X, y)
  return max(scores) # return best score from last cv

def reload():
  '''Development function to reload all task modules'''
  importlib.reload(A1)
  importlib.reload(A2)
  importlib.reload(B1)
  importlib.reload(B2)