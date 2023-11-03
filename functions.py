import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


# This function builds a model with given input_dimension (number of words in dictionary),
# output_dimension (number of classes), perceptrons list (where each number is the next layer),
# activation function, and learning rate.
def build_model(input_dimension=2, output_dimension=5, perceptrons=None, act='sigmoid', LR=0.2):
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


# This functions trains the given model
def fit_model(model, X, Y, epochs=300, validation_split=None):
    # fit the model (fit: returns history of training)
    history = model.fit(X, Y, epochs=epochs, verbose=2, validation_split=validation_split)

    # final prediction (after training)
    y_hat = model.predict(X).flatten()

    return [history, y_hat]


# This function generates plot of history of accuracy during training and/or validation.
def plot_loss_accuracy(history, validation=False):
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
    # plt.savefig("model1.svg")


# This function splits the dataset into X and Y matrices and returns them, as well as the label encoder for Y.
def split_xy(dataset):
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
