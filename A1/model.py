import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def load_image(filename):
    img = io.imread(filename, as_gray=True) # use built-in grayscale loading
    crop = img[90:180, 54:124] # crop to face
    return crop

def feature_extraction(X, test=False):
    global pca
    Xf = X.reshape(len(X),-1) # use this to flatten the last dims of X without copying
    if not test:
        pca = PCA(n_components=100).fit(Xf)
        plt.plot(pca.explained_variance_ratio_.cumsum())
        plt.show()
    return pca.transform(Xf)

def fit(X, y):
    global model
    param_grid = {'C':[0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10]}
    model = GridSearchCV(SVC(), param_grid, verbose=3).fit(X, y)
    return model.score(X, y)

def predict(X):
    global model
    return model.preict(X)
