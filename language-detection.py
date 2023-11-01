import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# https://www.kaggle.com/datasets/basilb2s/language-detection?resource=download
# dataset = pd.read_csv("/kaggle/input/language-detection/Language Detection.csv")
dataset = pd.read_csv("Language Detection.csv")

print(dataset.groupby(["Language"])["Text"].count().nlargest(17))

# I removed Hindi, because there is very little documents in the dataset.

print(dataset.isna().sum())

dataset = dataset[dataset['Language'] != 'Hindi']

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

# data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)


# ## Model 1 - Sigmoid activation

print("Sigmoid")


def build_model(input_dimension=2, output_dimension=5, perceptrons=[10], act='sigmoid', hidden_layers=1, LR=0.2):
    # define model
    model = tf.keras.models.Sequential()

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
    # fit the model (fit: returns history of training)
    history = model.fit(X, Y, epochs=epochs, verbose=2, validation_split=validation_split)

    # final prediction (after training)
    y_hat = model.predict(X).flatten()

    return [history, y_hat]


model1 = build_model(input_dimension=X.shape[1], output_dimension=16, perceptrons=[32],
                     act='sigmoid', hidden_layers=1, LR=0.002)


# ### Training

loss_accuracy = fit_model(model1, X_train, Y_train, epochs=5, validation_split=0.2)[0]


# ### Evaluation

Y_pred = model1.predict(X_test)
Y_pred_lab = lang_enc.inverse_transform(Y_pred)
Y_test_lab = lang_enc.inverse_transform(Y_test)

print(classification_report(Y_test_lab, Y_pred_lab))


# pd.DataFrame(classification_report(Y_test_lab, Y_pred_lab, output_dict=True)).T.round(2).to_latex()


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
    plt.savefig("model1.svg")


# PLOT TRAINING HISTORY
plot_loss_accuracy(loss_accuracy.history, validation=True)

# TEST THE MODEL
loss, accuracy = model1.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# ## Model 2 - ReLU activation

model2 = build_model(input_dimension=X.shape[1], output_dimension=16,
                     perceptrons=[64, 16], act='ReLU', hidden_layers=2, LR=0.0005)

# ### Training

loss_accuracy = fit_model(model2, X_train, Y_train, epochs=6, validation_split=0.2)[0]


# ### Evaluation

Y_pred = model2.predict(X_test)
Y_pred_lab = lang_enc.inverse_transform(Y_pred)
Y_test_lab = lang_enc.inverse_transform(Y_test)

print(classification_report(Y_test_lab, Y_pred_lab))

# pd.DataFrame(classification_report(Y_test_lab, Y_pred_lab, output_dict=True)).T.round(2).to_latex()

# PLOT TRAINING HISTORY
plot_loss_accuracy(loss_accuracy.history, validation=True)

# TEST THE MODEL
loss, accuracy = model2.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
