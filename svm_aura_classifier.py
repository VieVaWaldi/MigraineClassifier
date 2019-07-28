import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn import svm

KERNEL = 'poly'															# linear, rbf, poly
GAMMA = 100

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

    clf = svm.SVC(kernel=KERNEL, gamma=GAMMA)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def experiment_7f():
    """ wie 7c, alle age groups, einmal fuer maenner, einmal fuer frauen """
    print('############## Experiment 7f ##########################')

    for j in range(2):  # gender 0 (m) - 1 (f)
        print(
            '############## Experiment 7f Versuch {} ##########################'.format(j))
        # name = 'Exp_7f_gender={}hidden={}_drop=0.15'.format(j, hidden_layer)

        input_shape = 11

        dataset = pd.read_csv('data/processed/processed_scaled.csv')
        dataset = dataset.values

        Y = dataset[:, 22]

        X = dataset[:, 4:8]					      # painloc, -intensity, -origin, -type

        X = np.column_stack((X, dataset[:, 10]))  # licht
        X = np.column_stack((X, dataset[:, 13]))  # ruhe

        X = np.column_stack((X, dataset[:, 11]))  # laerm
        X = np.column_stack((X, dataset[:, 18]))  # med1
        X = np.column_stack((X, dataset[:, 19]))  # medeff

        X = np.column_stack((X, dataset[:, 20]))  # altersgruppe
        X = np.column_stack((X, dataset[:, 2]))   # gender

        i = 0
        for col in X:  # delete gender
            gender = col[10]
            if j == 0:
                if int(gender) == 1:
                    X = np.delete(X, obj=i, axis=0)
                    Y = np.delete(Y, obj=i, axis=0)
                    i -= 1
            if j == 1:
                if int(gender) == 0:
                    X = np.delete(X, obj=i, axis=0)
                    Y = np.delete(Y, obj=i, axis=0)
                    i -= 1
            i += 1

        X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=109)

        clf = svm.SVC(kernel=KERNEL, gamma=GAMMA)  
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
    # experiment_1()
    experiment_7f()
