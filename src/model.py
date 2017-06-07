#!/usr/bin/python

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, SReLU
from keras.callbacks import BaseLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import os

# Reproducible random seed
seed = 1
np.random.seed(seed)

# Create the tmp output directory, if it doesn't exist
try:
    os.makedirs("tmp")
except OSError:
    if not os.path.isdir("tmp"):
        raise

try:
    os.makedirs("logs")
except OSError:
    if not os.path.isdir("logs"):
        raise

# Import and normalize the data
data = pd.read_csv('data/creditcard.csv')
data.iloc[:, 1:29] = StandardScaler().fit_transform(data.iloc[:, 1:29])
data_matrix = data.as_matrix()
X = data_matrix[:, 1:29]
Y = data_matrix[:, 30]
class_weights = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], Y)))

# Create model with k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
predictions = np.empty(len(Y))
predictions[:] = np.NAN
proba = np.empty([len(Y), kfold.n_splits])
proba[:] = np.NAN
k = 0 
for train, test in kfold.split(X, Y):
    # Define model
    model = Sequential()
    model.add(Dense(28, input_dim=28))
    model.add(SReLU())
    model.add(Dropout(0.2))
    model.add(Dense(22))
    model.add(SReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Define callbacks
    baselogger = BaseLogger()
    checkpointer = ModelCheckpoint(filepath="tmp/weights.hdf5", verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    # Compile model
    metrics = ['binary_accuracy', 'fmeasure', 'precision', 'recall']
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    # Fit the model
    history = model.fit(X[train], Y[train],
                        batch_size=1200,
                        nb_epoch=100,
                        verbose=0,
                        shuffle=True,
                        validation_data=(X[test], Y[test]),
                        class_weight=class_weights,
                        callbacks=[baselogger, checkpointer, earlystop, reduce_lr, tensor_board])
    # Evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    # Store the predicted probabilities and iterate k
    proba[train, k] = model.predict_proba(X[train]).flatten()
    k += 1

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
pred = np.nanmean(proba, 1) > 0.5
pred = pred.astype(int)
print(classification_report(Y, pred))
print(pd.crosstab(Y, pred, rownames=['Truth'], colnames=['Predictions']))
