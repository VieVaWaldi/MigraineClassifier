import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# from keras.models import load_model
from keras import regularizers

import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 500

OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'
ACTIVATION = 'relu'
ACTIVATION_LAST_LAYER = 'sigmoid'

PATH = './experiments/ml_classifier/'


def experiment_1(hidden_layer):
    """ Alle Daten, keine Gruppen (group_time, group_age) """
    print('############## Experiment 1 ##########################')

    name = 'Exp_1_all_data_input=0:20_hidden={}_drop=0.15'.format(hidden_layer)
    input_shape = 20

    dataset = pd.read_csv('data/processed/processed_scaled.csv')
    dataset = dataset.values

    # Values 0 - 19, includes everything except group_time and group_age
    # The end-slice marks the border:
    # https://medium.com/@buch.willi/numpys-indexing-and-slicing-notation-explained-visually-67dc981c22c1
    X = dataset[:, 0:20]
    Y = dataset[:, 22]

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
        X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_val_and_test, Y_val_and_test, test_size=0.5)

    model = Sequential([
        Dense(
            hidden_layer, activation=ACTIVATION,
            input_shape=(input_shape,)),
        Dropout(0.15),
        Dense(1, activation=ACTIVATION_LAST_LAYER),
    ])

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

    hist = model.fit(
        X_train, Y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, validation_data=(X_val, Y_val))

    model.evaluate(X_test, Y_test)[1]

    # plot_acc(hist).show()
    model.save('{}exp_1/{}.h5'.format(PATH, name))
    plot_acc(hist).savefig(
        PATH + 'exp_1/plot_acc_hidden={}.png'.format(hidden_layer))
    plot_loss(hist).savefig(
        PATH + 'exp_1/plot_loss_hidden={}.png'.format(hidden_layer))


def experiment_2(hidden_layer):
    """ Alle Daten, Gruppen (group_time, group_age) anstatt von age, time"""
    print('############## Experiment 2 ##########################')

    name = 'Exp_2_all_data_input=2:22_hidden={}_drop=0.15'.format(hidden_layer)
    input_shape = 20

    dataset = pd.read_csv('data/processed/processed_scaled.csv')
    dataset = dataset.values

    # Values 2 - 22, includes everything except age and time
    X = dataset[:, 2:22]
    Y = dataset[:, 22]

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
        X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_val_and_test, Y_val_and_test, test_size=0.5)

    model = Sequential([
        Dense(
            hidden_layer, activation=ACTIVATION,
            input_shape=(input_shape,)),
        Dense(1, activation=ACTIVATION_LAST_LAYER),
        Dropout(0.15)
    ])

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

    hist = model.fit(
        X_train, Y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, validation_data=(X_val, Y_val))

    model.evaluate(X_test, Y_test)[1]

    # plot_acc(hist).show()
    model.save('{}exp_2/{}.h5'.format(PATH, name))
    plot_acc(hist).savefig(
        PATH + 'exp_2/plot_acc_hidden={}.png'.format(hidden_layer))
    plot_loss(hist).savefig(
        PATH + 'exp_2/plot_loss_hidden={}.png'.format(hidden_layer))


def experiment_3(hidden_layer):
    """ Every data once on its own """
    print('############## Experiment 3 ##########################')

    input_shape = 1

    dataset = pd.read_csv('data/processed/processed_scaled.csv')
    dataset = dataset.values

    Y = dataset[:, 22]

    for i in range(22):
        print('############## Experiment 3 Versuch {} ##########################'.format(i))
        name = 'Exp_3_every_data_once_input={}_hidden={}_drop=0.15'.format(
            i, hidden_layer)
        X = dataset[:, i]

        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
            X, Y, test_size=0.3)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_val_and_test, Y_val_and_test, test_size=0.5)

        model = Sequential([
            Dense(
                hidden_layer, activation=ACTIVATION,
                input_shape=(input_shape,)),
            Dense(1, activation=ACTIVATION_LAST_LAYER),
            Dropout(0.15)
        ])

        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

        hist = model.fit(
            X_train, Y_train, batch_size=BATCH_SIZE,
            epochs=300, validation_data=(X_val, Y_val))

        model.evaluate(X_test, Y_test)[1]

        # plot_acc(hist).show()
        model.save('{}exp_3/{}.h5'.format(PATH, name))
        plot_acc(hist).savefig(
            PATH + 'exp_3/plot_acc_data={}_hidden={}.png'.format(i, hidden_layer))
        plot_loss(hist).savefig(
            PATH + 'exp_3/plot_loss_data={}_hidden={}.png'.format(i, hidden_layer))


def experiment_4(hidden_layer):
    """ Alle Daten, Gruppen (group_time, group_age) anstatt von age, time"""
    print('############## Experiment 4 ##########################')

    name = 'Exp_4_male_only_input=0:20_hidden={}_drop=0.15'.format(
        hidden_layer)
    input_shape = 20

    dataset = pd.read_csv('data/processed/processed_scaled.csv')
    dataset = dataset.values

    # Values 2 - 22, includes everything except age and time
    X = dataset[:, 0:20]
    Y = dataset[:, 22]

    i = 0
    for col in X:
        gender = col[2]
        if int(gender) is not 0:
            X = np.delete(X, obj=i, axis=0)
            Y = np.delete(Y, obj=i, axis=0)
            i -= 1
        i += 1

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
        X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_val_and_test, Y_val_and_test, test_size=0.5)

    model = Sequential([
        Dense(
            hidden_layer, activation=ACTIVATION,
            input_shape=(input_shape,)),
        Dense(1, activation=ACTIVATION_LAST_LAYER),
        Dropout(0.15)
    ])

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

    hist = model.fit(
        X_train, Y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, validation_data=(X_val, Y_val))

    model.evaluate(X_test, Y_test)[1]

    # plot_acc(hist).show()
    model.save('{}exp_4/{}.h5'.format(PATH, name))
    plot_acc(hist).savefig(
        PATH + 'exp_4/plot_acc_hidden={}.png'.format(hidden_layer))
    plot_loss(hist).savefig(
        PATH + 'exp_4/plot_loss_hidden={}.png'.format(hidden_layer))


def experiment_5(hidden_layer):
    """ Alle Daten, Gruppen (group_time, group_age) anstatt von age, time"""
    print('############## Experiment 5 ##########################')

    name = 'Exp_5_female_only_input=0:20_hidden={}_drop=0.15'.format(
        hidden_layer)
    input_shape = 20

    dataset = pd.read_csv('data/processed/processed_scaled.csv')
    dataset = dataset.values

    # Values 2 - 22, includes everything except age and time
    X = dataset[:, 0:20]
    Y = dataset[:, 22]

    i = 0
    for col in X:
        gender = col[2]
        if int(gender) is 0:
            X = np.delete(X, obj=i, axis=0)
            Y = np.delete(Y, obj=i, axis=0)
            i -= 1
        i += 1

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
        X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_val_and_test, Y_val_and_test, test_size=0.5)

    model = Sequential([
        Dense(
            hidden_layer, activation=ACTIVATION,
            input_shape=(input_shape,)),
        Dense(1, activation=ACTIVATION_LAST_LAYER),
        Dropout(0.15)
    ])

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

    hist = model.fit(
        X_train, Y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, validation_data=(X_val, Y_val))

    model.evaluate(X_test, Y_test)[1]

    # plot_acc(hist).show()
    model.save('{}exp_5/{}.h5'.format(PATH, name))
    plot_acc(hist).savefig(
        PATH + 'exp_5/plot_acc_hidden={}.png'.format(hidden_layer))
    plot_loss(hist).savefig(
        PATH + 'exp_5/plot_loss_hidden={}.png'.format(hidden_layer))


def experiment_6(hidden_layer):
    """ Only Pain parameters, type, origin, location, intensity """
    print('############## Experiment 6 ##########################')

    name = 'Exp_6_pain_para_only_input=4:8_hidden={}_drop=0.15'.format(hidden_layer)
    input_shape = 4

    dataset = pd.read_csv('data/processed/processed_scaled.csv')
    dataset = dataset.values

    X = dataset[:, 4:8]
    Y = dataset[:, 22]

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
        X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_val_and_test, Y_val_and_test, test_size=0.5)

    model = Sequential([
        Dense(
            hidden_layer, activation=ACTIVATION,
            input_shape=(input_shape,)),
        Dropout(0.15),
        Dense(1, activation=ACTIVATION_LAST_LAYER),
    ])

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

    hist = model.fit(
        X_train, Y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, validation_data=(X_val, Y_val))

    model.evaluate(X_test, Y_test)[1]

    # plot_acc(hist).show()
    model.save('{}exp_6/{}.h5'.format(PATH, name))
    plot_acc(hist).savefig(
        PATH + 'exp_6/plot_acc_hidden={}.png'.format(hidden_layer))
    plot_loss(hist).savefig(
        PATH + 'exp_6/plot_loss_hidden={}.png'.format(hidden_layer))


def experiment_7a(hidden_layer):
    """ Best 3 (Intensity 4, Licht 10, Ruhe 13), (plus Pain parameters) """
    print('############## Experiment 7a ##########################')

    name = 'Exp_7a_best_4_PLUS_PAIN_input=0:20_hidden={}_drop=0.15'.format(hidden_layer)
    input_shape = 4

    dataset = pd.read_csv('data/processed/processed_scaled.csv')
    dataset = dataset.values

    X = dataset[:, 4:8]
    Y = dataset[:, 22]

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
        X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_val_and_test, Y_val_and_test, test_size=0.5)

    model = Sequential([
        Dense(
            hidden_layer, activation=ACTIVATION,
            input_shape=(input_shape,)),
        Dropout(0.15),
        Dense(1, activation=ACTIVATION_LAST_LAYER),
    ])

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

    hist = model.fit(
        X_train, Y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, validation_data=(X_val, Y_val))

    model.evaluate(X_test, Y_test)[1]

    # plot_acc(hist).show()
    model.save('{}exp_6/{}.h5'.format(PATH, name))
    plot_acc(hist).savefig(
        PATH + 'exp_6/plot_acc_hidden={}.png'.format(hidden_layer))
    plot_loss(hist).savefig(
        PATH + 'exp_6/plot_loss_hidden={}.png'.format(hidden_layer))


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


def np_debug():
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)


if __name__ == '__main__':
    # experiment_1(64)
    # experiment_1(256)
    # experiment_1(512)

    # experiment_2(64)
    # experiment_2(256)
    # experiment_2(512)

    # experiment_3(64)
    # experiment_3(256)
    # experiment_3(512)

    # experiment_4(64)
    # experiment_4(256)
    # experiment_4(512)

    # experiment_5(64)
    # experiment_5(256)
    # experiment_5(512)

    experiment_6(512)
