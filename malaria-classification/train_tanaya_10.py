import numpy as np
from matplotlib import pyplot
from keras.models import load_model
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import keras
from keras.callbacks import ModelCheckpoint
from predict_tanaya_10 import predict
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
import os


class MyMLP:

    def __init__(self):
        self.features, self.label = [], []
        self.DATA_DIR = os.getcwd() + "/train/"
        self.RESIZE_TO = 50
        self.SEED = 0
        self.N_EPOCHS = 20
        self.BATCH_SIZE = 50
        self.LR = 1e-4


class DataProcessing(MyMLP):
    def __init__(self):
        super().__init__()

    def read_data(self, x):
        """
        Read images and augment the data by various transformations such as flip, blur, brightness, and laplacian

        :param x: list of path of images
        :return: numpy array of features and labels
        """
        for path in [f for f in x if f[-4:] == ".png"]:
            image = cv2.resize(cv2.imread(self.DATA_DIR + path), (self.RESIZE_TO, self.RESIZE_TO))
            flip_1 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            self.features.append(image)
            self.features.append(flip_1)
            with open(self.DATA_DIR + path[:-4] + ".txt", "r") as s:
                y = s.read()
            self.label.append(y)
            self.label.append(y)

            if y == "trophozoite":
                self.get_augmented_data(image, y, brightness=True)
            if y == "ring":
                self.get_augmented_data(image, y, brightness=True, blur=True, laplacian=True)
            if y == "schizont":
                self.get_augmented_data(image, y, brightness=True, blur=True, laplacian=True)

        features, label = np.array(self.features), np.array(self.label)
        print("Size", features.shape, label.shape)
        return features, label

    def get_augmented_data(self, image, y, brightness=False, blur=False, laplacian=False):
        """
        Transform the images depending on the flags for brightness, blur or laplacian
        :param image: pixels of an image
        :param y: lable of the corresponding image
        :param brightness: Boolean flag
        :param blur: Boolean flag
        :param laplacian: Boolean flag
        :return: append features and label list according to the corresponding transformation
        """

        if brightness:
            brightness = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            self.features.append(brightness)
            self.label.append(y)

        if blur:
            blur = cv2.medianBlur(image, 5)
            self.features.append(blur)
            self.label.append(y)

        if laplacian:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            self.features.append(laplacian)
            self.label.append(y)

    def transform_label(self, label):
        """

        :param label:
        :return:
        """
        le = LabelEncoder()
        le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
        label = le.transform(label)

        # convert class vectors to binary class matrices
        label = keras.utils.to_categorical(label, 4)

        print("Read data", features.shape, label.shape)
        return label

    def plot_images(self, features, label):
        """

        :param features:
        :param label:
        :return:
        """
        CLASSES = ["red blood cell", "ring", "schizont", "trophozoite"]
        for cl in CLASSES:
            ring = np.where(label == cl)
            for i in range(0, 9):
                pyplot.subplot(330 + 1 + i)
                pyplot.imshow(features[ring[0][i]].reshape(len(features[i]), -1))
                # show the plot
            pyplot.show()


class ModelBuilding(MyMLP):
    def __init__(self):
        super().__init__()

    def train_model(self, features, label):
        """

        :param features:
        :param label:
        :return:
        """
        x, x_test, y, y_test = train_test_split(features, label, random_state=self.SEED, test_size=0.2,
                                                stratify=label)
        print("x, y", x.shape, y.shape)
        np.save('x_test.npy', x_test)
        np.save('y_test.npy', y_test)

        x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=self.SEED, test_size=0.2,
                                                          stratify=y)

        model = self.create_model()
        # model = get_grid(x,y)

        # model.fit(x_train, y_train, epochs=15, verbose=1, validation_data=[x_val, y_val], batch_size=BATCH_SIZE,
        #           callbacks=[
        #               ModelCheckpoint("./model/model_16/mlp_tanaya_10_7.hdf5", monitor="val_loss", save_best_only=True)])
        model = load_model("./model/model_16/mlp_tanaya_10_4.hdf5")
        #
        print("Final accuracy on validations set:", 100 * model.evaluate(x_test, y_test)[1], "%")
        print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1)))
        print("F1 score",
              f1_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average='macro'))
        print("Confusion Matrix---->")
        print(
            confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1), labels=[0, 1, 2, 3]))

        return model

    def create_model(self):
        """

        :return:
        """
        model = Sequential()
        model.add(Dense(350, activation="relu", input_shape=(50, 50, 3)))
        model.add(Dense(400, activation="relu"))
        model.add(Dense(500, activation="relu"))
        model.add(Dense(700, activation="relu"))
        # model.add(Dropout(DROPOUT))
        model.add(Dense(900, activation="relu"))
        # model.add(Dropout(DROPOUT))
        model.add(Dense(700, activation="relu"))
        model.add(Dense(500, activation="relu"))
        model.add(Dense(400, activation="relu"))
        # model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(4, activation="softmax"))
        model.summary()

        model.compile(optimizer=Adam(lr=self.LR), loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def get_grid(self, x, y):
        """
        
        :param x:
        :param y:
        :return:
        """
        # define the grid search parameters
        optimizer = ['SGD', 'Adam', 'Adamax']
        init_mode = ['uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform']

        param_grid = dict(epochs=[20], init_mode=init_mode)
        model = KerasClassifier(build_fn=self.create_model, batch_size=20)
        # history = LossHistory()
        #
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
        # # callbacks = [ModelCheckpoint("mlp_tanaya_10_gs.hdf5", monitor="val_loss", save_best_only=True)]

        grid_result = grid.fit(x, y, callbacks=[
            ModelCheckpoint("mlp_tanaya_10_gs.hdf5", monitor="val_loss", save_best_only=True)])
        print("grid", grid_result.best_estimator_)
        model = grid_result.best_estimator_
        # # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        y_pred = model.predict(np.load('x_test.npy'))
        print(y_pred)
        print("unique", len(np.unique(y_pred)))

        y_test = np.argmax(np.load('y_test.npy'), axis=1)
        print(y_test)

        # %% ------------------------------------------ Final test# -------------------------------------------------------------
        # print("Final accuracy on validations set:", 100 * grid_result.evaluate(x_test, y_test)[1], "%")
        print("Cohen Kappa", cohen_kappa_score(y_pred, y_test))
        print("F1 score",
              f1_score(y_pred, y_test, average='macro'))

        return model


mpl_obj = DataProcessing()
x = os.listdir(mpl_obj.DATA_DIR)
features, label = mpl_obj.read_data(x)
label = mpl_obj.transform_label(label)
features = features / 255

model_obj = ModelBuilding()
model = model_obj.train_model(features, label)



