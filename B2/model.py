import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn import tree

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
    img = cv2.imread(filename) # this task needs colour
    return img[240:280,160:340,::-1]/255.0 # crop to eyes, reverse GBR to RGB

def fit(X, y):
    global model
    X = X.reshape(len(X),-1) # flatten the last dims of X without copying
    model = tree.DecisionTreeClassifier(max_depth=8, random_state=RND).fit(X, y)
    plt.figure(figsize=(12,10))
    tree.plot_tree(model, fontsize=9) # plot the decision tree
    plt.show()
    return model

def predict(X):
    global model
    X = X.reshape(len(X),-1) # flatten the last dims of X without copying
    return model.predict(X) # select prediction with highest output
