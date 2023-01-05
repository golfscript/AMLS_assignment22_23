import numpy as np
from os import path
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

LABELS = 'labels.csv'
DATASETS = 'Datasets'
IMGS = 'img'
TEST = '_test'

def load_data(folder, feature_col, file_col, load_image, test=False, augment=False):
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
  X = np.empty((n*2 if augment else n, *shape), dtype=np.uint8) # pre-define X, much more efficient than concatenating arrays
  for i in tqdm(range(n), desc=folder): # tqdm displays a nice loading bar
      X[i] = load_image(path.join(imagedir,filenames[i]))
  if augment:
      X[n:] = X[:n,:,::-1]  # copy images, but reverse left to right
      y = np.tile(y,2) # duplicate labels
  print(f'Loaded {X.nbytes:,} bytes')
  show_images(X[:5],y[:5])
  return X, y

def show_images(X, y):
  plt.figure(figsize=(12,3))
  for i in range(len(X)):
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
