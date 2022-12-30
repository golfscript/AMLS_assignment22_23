import numpy as np
from matplotlib import pyplot as plt
from skimage import io, exposure
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

def load_image(filename):
    img = io.imread(filename, as_gray=True) # use built-in grayscale loading
    crop = img[90:180, 54:124] # crop to face
    return exposure.equalize_hist(crop)

def fit(X, y):
    global pca,model
    Xf = X.reshape(len(X),-1) # use this to flatten the last dims of X without copying
    pca = PCA(n_components=100).fit(Xf)
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.show()
    Xp = pca.transform(Xf)
    param_grid = {'kernel':['rbf','sigmoid'], 'C':[0.5, 1, 2, 5, 10], 'gamma':[0.005, 0.01, 0.02, 0.05, 0.1]}
    model = HalvingGridSearchCV(SVC(), param_grid, verbose=3).fit(Xp, y)
    print(model.best_params_)
    return model.best_score_

def predict(X):
    global pca,model
    Xf = X.reshape(len(X),-1)
    return model.predict(pca.transform(Xf))
