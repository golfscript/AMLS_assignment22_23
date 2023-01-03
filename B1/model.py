import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn import tree

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # use built-in grayscale conversion
    crop = img[160:400,150:350] # crop to face
    scale = cv2.resize(crop,None,fx=1/2,fy=1/2,interpolation=cv2.INTER_NEAREST_EXACT) # rescale to one third size
    return scale

def fit(X, y):
    global model
    X = X.reshape(len(X),-1) # flatten the last dims of X without copying
    model = tree.DecisionTreeClassifier(max_depth=8, random_state=RND).fit(X, y)
    plt.figure(figsize=(12,10))
    tree.plot_tree(model, fontsize=9) # plot the decision tree
    plt.show()
    return model.score(X,y)

def predict(X):
    global model
    X = X.reshape(len(X),-1) # flatten the last dims of X without copying
    return model.predict(X) # select prediction with highest output
