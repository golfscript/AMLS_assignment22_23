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
    return exposure.equalize_adapthist(crop/255.0)*255 # apply adaptive histogram equalisation

def fit(X, y):
    global pca,model
    X = X.reshape(len(X),-1) # flatten the last dims of X without copying
    X = np.divide(X,255,dtype=np.float32) # convert to [0,1] and limit to float32 to save space
    print('Performing Principle Component Analysis')
    pca = PCA(n_components=120, random_state=RND)
    X = pca.fit_transform(X) # do fit & transform in one step as quicker
    plt.title('Cumulative sum of variance explanied by #components')
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

    print('Performing Halving Grid Search with Cross Validation')
    Cs = [0.5, 1, 2, 5, 10]
    param_grid = [{'kernel':['linear','poly'], 'C':Cs},
                  {'kernel': ['rbf','sigmoid'], 'C':Cs, 'gamma':[0.001, 0.003, 0.01, 0.03, 0.1]}]
    model = HalvingGridSearchCV(SVC(), param_grid, max_resources=len(X)//2, random_state=RND, verbose=3).fit(X, y)
    print("Optimal hyper-parameters:", model.best_params_)
    print("Mean cross-validated accuracy:", model.best_score_)
    return model.score(X,y)

def predict(X):
    global pca,model
    X = X.reshape(len(X),-1) # flatten the last dims of X without copying
    X = np.divide(X,255,dtype=np.float32) # limit to float32 to save space
    X = pca.transform(X)
    return model.predict(X)
