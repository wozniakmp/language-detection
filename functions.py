import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


def build_model(input_dimension=2, output_dimension=5, perceptrons=None, act='sigmoid', LR=0.2):
    """
    This function builds a model with given parameters.
    :param input_dimension: number of words in dictionary
    :param output_dimension: number of classes
    :param perceptrons: list of perceptrons in each layer
    :param act: activation function ('sigmoid' by default)
    :param LR: Learning Rate (0.2 by default)
    :return: compiled model
    """
    if perceptrons is None:
        perceptrons = [8]
    # define model
    model = tf.keras.models.Sequential()
    hidden_layers = len(perceptrons)

    # add layers
    # Dense(perceptrons, input_dim (features), activation (nonlinear function))
    model.add(tf.keras.layers.Dense(perceptrons[0], input_dim=input_dimension, activation=act))  # hidden layer 1
    for i in range(hidden_layers-1):
        model.add(tf.keras.layers.Dense(perceptrons[i+1], activation=act))  # additional hidden layers
    model.add(tf.keras.layers.Dense(output_dimension, activation='softmax'))  # output layer
    model.summary()

    # compile model
    my_opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(loss='binary_crossentropy', optimizer=my_opt, metrics=['accuracy'])

    return model


def fit_model(model, X, Y, epochs=300, validation_split=None):
    """
    Function that fits the given model to the given data.
    :param model: tk.keras model to fit
    :param X: matrix of input data
    :param Y: vector of output labels
    :param epochs: number of epochs
    :param validation_split: size of the validation set (between 0-1)
    :return: history of training and model's predictions
    """
    # fit the model (fit: returns history of training)
    history = model.fit(X, Y, epochs=epochs, verbose=2, validation_split=validation_split)

    # final prediction (after training)
    y_hat = model.predict(X).flatten()

    return [history, y_hat]


def plot_loss_accuracy(history, validation=False):
    """
    This function generates plot of history of accuracy during training and/or validation.
    :param history: history of training
    :param validation: boolean (False by default)
    :return: nothing (draws a plot)
    """
    loss = history['loss']
    acc = history['accuracy']
    legend = ['train']
    if validation:
        legend.append('validation')
        val_loss = history['val_loss']
        val_acc = history['val_accuracy']

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Log Loss and Accuracy over iterations')

    # add_subplot(rows, columns, index)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss)
    if validation:
        ax.plot(val_loss)
    ax.grid(True)
    ax.set(xlabel='epochs', title='Log Loss')
    ax.legend(legend, loc='upper right')

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(acc)
    if validation:
        ax.plot(val_acc)
    ax.grid(True)
    ax.set(xlabel='epochs', title='Accuracy')
    ax.legend(legend, loc='lower right')
    plt.show()
    # plt.savefig("model1.svg")


def split_xy(dataset):
    """
    This function splits the dataset into X and Y matrices and returns them,
    as well as the label encoder for Y.
    :param dataset: training set
    :return: input data, output data, label encoder
    """
    text = dataset['Text']
    lang = dataset['Language']

    # transform data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text).toarray()
    print(X.shape)

    # transform labels
    lang = np.array(lang).reshape(-1, 1)
    lang_enc = OneHotEncoder().fit(lang)
    Y = lang_enc.transform(lang).toarray()

    return X, Y, lang_enc
