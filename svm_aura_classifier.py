# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn import svm

NAME = ''
KERNEL = 'poly'															# linear, rfb, poly
GAMMA = 0.01
PROBABILITY = False

# obj 				is the optimal objective value of the dual SVM problem
# rho 				is the bias term in the decision function sgn(w^Tx - rho)
# nSV and nBSV 		are number of support vectors and bounded support vectors (i.e., alpha_i = C)
# nu-svm 			is a somewhat equivalent form of C-SVM where C is replaced by nu
# nu 				simply shows the corresponding parameter. More details are in libsvm document


def experiment_1():
    """ Alle Daten, keine Gruppen (group_time, group_age) """
    name = 'Exp_1_all_data_input=0:20'

    dataset = pd.read_csv('data/processed/processed_scaled.csv')
    dataset = dataset.values

    # Values 0 - 19, includes everything except group_time and group_age
    X = dataset[:, 0:20]
    Y = dataset[:, 22]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=109)

    clf = svm.SVC(kernel='rbf')  # Linear Kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def plot_loss(hist):
    plt.clf()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    return plt


def plot_acc(hist):
    plt.clf()
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    return plt


if __name__ == '__main__':
    experiment_1()
