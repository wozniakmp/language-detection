# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# import tensorflow as tf
import functions as fn
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# https://www.kaggle.com/datasets/basilb2s/language-detection?resource=download
# dataset = pd.read_csv("/kaggle/input/language-detection/Language Detection.csv")
dataset = pd.read_csv("Language Detection.csv")

print(dataset.groupby(["Language"])["Text"].count().nlargest(17))

# I removed Hindi, because there is very little documents in the dataset.

print(dataset.isna().sum())

dataset = dataset[dataset['Language'] != 'Hindi']

X, Y, lang_enc = fn.split_xy(dataset)

# data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)

# ## Model 1 - Sigmoid activation

print("Sigmoid")

model1 = fn.build_model(input_dimension=X.shape[1], output_dimension=16, perceptrons=[32],
                        act='sigmoid', LR=0.002)

# ### Training

loss_accuracy = fn.fit_model(model1, X_train, Y_train, epochs=5, validation_split=0.2)[0]

# ### Evaluation

Y_pred = model1.predict(X_test)
Y_pred_lab = lang_enc.inverse_transform(Y_pred)
Y_test_lab = lang_enc.inverse_transform(Y_test)

print(classification_report(Y_test_lab, Y_pred_lab))

# pd.DataFrame(classification_report(Y_test_lab, Y_pred_lab, output_dict=True)).T.round(2).to_latex()

# PLOT TRAINING HISTORY
fn.plot_loss_accuracy(loss_accuracy.history, validation=True)

# TEST THE MODEL
loss, accuracy = model1.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# ## Model 2 - ReLU activation

model2 = fn.build_model(input_dimension=X.shape[1], output_dimension=16,
                        perceptrons=[64, 16], act='ReLU', LR=0.0005)

# ### Training

loss_accuracy = fn.fit_model(model2, X_train, Y_train, epochs=6, validation_split=0.2)[0]

# ### Evaluation

Y_pred = model2.predict(X_test)
Y_pred_lab = lang_enc.inverse_transform(Y_pred)
Y_test_lab = lang_enc.inverse_transform(Y_test)

print(classification_report(Y_test_lab, Y_pred_lab))

# pd.DataFrame(classification_report(Y_test_lab, Y_pred_lab, output_dict=True)).T.round(2).to_latex()

# PLOT TRAINING HISTORY
fn.plot_loss_accuracy(loss_accuracy.history, validation=True)

# TEST THE MODEL
loss, accuracy = model2.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
