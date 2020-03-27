import numpy as np
from keras.models import load_model
import cv2
from sklearn.preprocessing import LabelEncoder
import keras

import os

# os.system("sudo pip install numpy")
# os.system("sudo pip install keras")
# os.system("sudo pip install cv2")
# os.system("sudo pip install sklearn")

# if "train" not in os.listdir():
#     os.system("wget https://storage.googleapis.com/exam-deep-learning/train.zip")
#     os.system("unzip train.zip")
#
# DATA_DIR = os.getcwd() + "/train/"
# RESIZE_TO = 50


def predict(x):
    # %% --------------------------------------------- Data Read -------------------------------------------------------
    features, label = [], []

    for path in [f for f in x if f[-4:] == ".png"]:
        features.append(cv2.resize(cv2.imread(path), (50, 50)))

    features, label = np.array(features), np.array(label)

    # Write any data prep you used during training
    # standardize features
    features = features / 255


    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_tanaya_10.hdf5')

    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5", etc.
    y_pred = np.argmax(model.predict(features), axis=1)

    return y_pred, model
    # If using more than one model to get y_pred, do the following:
    # return y_pred, model1, model2  # If you used two models
    # return y_pred, model1, model2, model3  # If you used three models, etc.



