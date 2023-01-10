import numpy as np
from os import path
from sklearn.model_selection import cross_val_score
import A1.models as A1, A2.models as A2, B1.models as B1, B2.models as B2
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import ipywidgets as widgets

LABELS = 'labels.csv'
DATASETS = 'Datasets'
IMGS = 'img'
TEST = '_test'

TASKS = {'A1':('celeba',2,1,A1), 'A2':('celeba',3,1,A2), 'B1':('cartoon_set',2,3,B1),'B2':('cartoon_set',1,3,B2)}
task_options = widgets.RadioButtons(options=TASKS, description='Task')

def load_data(folder, feature_col, file_col, load_image, test=False):
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

def show_images(X, y):
  plt.figure(figsize=(12,3))
  for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.title(f'[{i}] class:{y[i]}')
    plt.imshow(X[i], cmap='gray')
  plt.show()

def show_wrong(X, y, y_pred):
  wrong, = np.nonzero(y!=y_pred)
  print(f'There were {len(wrong)} wrong predictions out of {len(y)}')
  for w,i in enumerate(wrong):
    if w%5==0: plt.figure(figsize=(12,3))
    plt.subplot(1, 5, w%5+1)
    plt.axis('off')
    plt.title(f'[{i}] class:{y[i]} pred:{y_pred[i]}')
    plt.imshow(X[i], cmap='gray')
    if w%5==4: plt.show()

def cv_optimiser(model, X, y, params):
  for param, values in params.items():
    print(f'Peforming Cross Validation on optimal {param}...')
    prog_bar = tqdm(values, desc='cross validation')
    scores = [cross_val_score(model.set_params(**{param:v}), X, y, n_jobs=-1).mean() for v in prog_bar]
    plt.plot([str(v) for v in values], scores)
    plt.show()

    best = values[np.argmax(scores)]
    print(f'Optimal {param} is', best)
    model.set_params(**{param:best})

  print('Performing final fit on all data with optimal params...')
  return model.fit(X, y)
