import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv # need this to enable line below
from sklearn.model_selection import HalvingGridSearchCV

RND = 1 # use in all functions that need random seed in order to ensure repeatability
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(7,7))

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
    crop = img[90:180, 54:124] # crop to face
    return exposure.equalize_adapthist(crop/255.0)*255 # apply adaptive histogram equalisation

def fit(X, y):
    global pca,model
    X = X.reshape(len(X),-1) # flatten the last dims of X without copying
    Xf = np.divide(X,255,dtype=np.float32) # limit to float32 to save space
    pca = PCA(n_components=120, random_state=RND)
    Xp = pca.fit_transform(Xf)
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.show()

    param_grid = {'kernel':['rbf','sigmoid','linear'], 'C':[0.5, 1, 2, 5, 10], 'gamma':[0.001, 0.003, 0.01, 0.03, 0.1]}
    model = HalvingGridSearchCV(SVC(), param_grid, max_resources=len(X)//2, random_state=RND, verbose=3).fit(Xp, y)
    print("Optimal hyper-parameters:", model.best_params_)
    print("Mean cross-validated accuracy:", model.best_score_)
    return model.score(Xp,y)

def predict(X):
    global pca,model
    X = X.reshape(len(X),-1) # flatten the last dims of X without copying
    Xf = np.divide(X,255,dtype=np.float32) # limit to float32 to save space
    Xp = pca.transform(Xf)
    return model.predict(Xp)
