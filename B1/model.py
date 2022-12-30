import numpy as np
from matplotlib import pyplot as plt
from skimage import io, transform
from sklearn import tree

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
    img = io.imread(filename, as_gray=True) # use built-in grayscale loading
    crop = img[160:400,150:350] # crop to face
    return transform.rescale(crop,0.5) # rescale to half size

def fit(X, y):
    global model
    X = X.reshape(len(X),-1) # use this to flatten the last dims of X without copying
    model = tree.DecisionTreeClassifier(max_depth=8, random_state=RND).fit(X, y)
    plt.figure(figsize=(12,10))
    tree.plot_tree(model, fontsize=10) # plot the decision tree
    plt.show()
    return model.score(X,y)

def predict(X):
    global model
    X = X.reshape(len(X),-1) # use this to flatten the last dims of X without copying
    return model.predict(X) # select prediction with highest output
