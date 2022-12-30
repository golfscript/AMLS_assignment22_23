import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from sklearn import tree

RND = 1 # use in all functions that need random seed in order to ensure repeatability

def load_image(filename):
    img = ImageOps.grayscale(Image.open(filename)) # use built-in grayscale conversion
    crop = img.crop((150,160,350,400)) # crop to face
    scale = ImageOps.scale(crop,1/3,Image.NEAREST) # rescale to one third size
    return np.asarray(scale)/255.0 # convert to range [0,1]

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
