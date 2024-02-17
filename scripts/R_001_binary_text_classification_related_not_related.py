# -*- coding: utf-8 -*-
"""
R_001_Research_Project__
|
R_001_binary_text_classification_related_not_related.py
Created on Thu Jan 18 16:30:03 2024
@author: Rochana Obadage
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_core as keras
import keras_nlp
import pytz
import re

import argparse
import sys
import glob
import os


training_attempt = "009"
plot_path = f"plots/{training_attempt}"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)


df_train = pd.read_csv("dataset/binary_train.csv")
df_test = pd.read_csv("dataset/binary_test.csv")

print('Training Set Shape = {}'.format(df_train.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
print('Test Set Shape = {}'.format(df_test.shape))
print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))


BATCH_SIZE = 32
NUM_TRAINING_EXAMPLES = df_train.shape[0]
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.1
STEPS_PER_EPOCH = int(NUM_TRAINING_EXAMPLES)*TRAIN_SPLIT // BATCH_SIZE

EPOCHS = 12 #15, 20
AUTO = tf.data.experimental.AUTOTUNE

from sklearn.model_selection import train_test_split

X = df_train["context"]
y = df_train["target"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42, stratify=df_train.label.values)

X_test = df_test["context"]
y_test = df_test["target"]

# Load a DistilBERT model.
preset= "distil_bert_base_en_uncased"

# Use a shorter sequence length.
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(preset,
                                                                   sequence_length=512,
                                                                   )
# preprocessor_4_tweets
# Pretrained classifier.
classifier = keras_nlp.models.DistilBertClassifier.from_preset(preset,
                                                               preprocessor = preprocessor,
                                                               num_classes=2)

classifier.summary()

classifier.fit(x=X_train, y=y_train, batch_size=2)

# Re-compile 
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(1e-5),
    metrics= ["accuracy"],
)


history = classifier.fit(x=X_train,
                         y=y_train,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS,
                         validation_data=(X_val, y_val),
                        )

saved_model_path = f'lemos_R_001_binary_label_predictor_{training_attempt}_{EPOCHS}_ep.keras'
classifier.save(saved_model_path)

classifier.save(saved_model_path)

print("Training Completed")
print("Model saved")


def plot_confusion_matrix(y_true, y_preds, labels=None, type_="Test",normalize_="true"):
    cm = confusion_matrix(y_true, y_preds)
    cm = confusion_matrix(y_true, y_preds, normalize=normalize_)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    global EPOCHS
    plt.title(f"{type_} confusion matrix")
    plt.savefig(f'plots/{training_attempt}/{type_}ep_{EPOCHS}.png')
    # plt.show()


y_pred_train = classifier.predict(X_train)
y_pred_val = classifier.predict(X_val)
y_pred_test = classifier.predict(X_test)


label_to_score = {'related':1,"not_related":0}

with open('binary_model_perfromances.txt','a') as f:
    f.write(saved_model_path)
    f.write(f'_______{training_attempt} - epochs: {EPOCHS}_______\n')

    # training
    plot_confusion_matrix(y_train, np.argmax(y_pred_train, axis=1),labels= label_to_score.keys(),type_="Training",normalize_=None)  #normalize_="true" -- default
    print('classifiation report : Training')
    print(classification_report(y_train, np.argmax(y_pred_train, axis=1),target_names= label_to_score.keys()))
    
    f.write('classifiation report : Training\n')
    f.write(classification_report(y_train, np.argmax(y_pred_train, axis=1),target_names= label_to_score.keys()))
    f.write('\n\n')
    
    # # validation : confusion-matrix
    # validation
    plot_confusion_matrix(y_val, np.argmax(y_pred_val, axis=1),labels= label_to_score.keys(),type_="Validation",normalize_=None)
    print('classifiation report: Validation')
    print(classification_report(y_val, np.argmax(y_pred_val, axis=1),target_names= label_to_score.keys()))

    f.write('classifiation report : Validation\n')
    f.write(classification_report(y_val, np.argmax(y_pred_val, axis=1),target_names= label_to_score.keys()))
    f.write('\n\n')
    
    
    # # Testing : confusion-matrix
    # Testing
    plot_confusion_matrix(y_test, np.argmax(y_pred_test, axis=1),labels= label_to_score.keys(),type_="Testing",normalize_=None)
    print('classifiation report: Testing')
    print(classification_report(y_test, np.argmax(y_pred_test, axis=1),target_names= label_to_score.keys()))

    f.write('classifiation report : Testing\n')
    f.write(classification_report(y_test, np.argmax(y_pred_test, axis=1),target_names= label_to_score.keys()))
    f.write('\n\n\n\n')

