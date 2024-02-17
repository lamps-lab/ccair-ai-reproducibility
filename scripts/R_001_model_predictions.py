# -*- coding: utf-8 -*-
"""
R_001_Research_Project__
|
R_001_model_predictions.py
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

mapping = {'O-NR':0, 'P-NR':1, 'Neutral':2, 'Weak':3, 'Strong':4}
mapping_reverse = {0: 'O-NR', 1: 'P-NR', 2: 'Neutral', 3: 'Weak', 4: 'Strong'}
label_to_score = {'O-NR': -2, 'P-NR': -1, 'Neutral': 0, 'Weak': 0.5, 'Strong': 1}

saved_model_path = 'lemos_R_001_label_predictor_005_30ep_bert.keras'

model_path = 'lemos_R_001_label_predictor_005_30ep_bert.keras'
reloaded_model_22 = keras.models.load_model(model_path)

print("Model loaded")


zero_citation_list = ["RS_007","RS_070","RS_146"]
label_file_list = [i for i in glob.glob("automated_labelling/*.csv") if i.replace("automated_labelling/","")[:6] not in zero_citation_list]

print(len(label_file_list))

for file_ in label_file_list:
    df_ = pd.read_csv(file_)
    print(len(df_))

    predictions_df = df_.copy()

    predictions_df['target_predict'] = 0
    x_test_series = predictions_df['text']

    predictions = reloaded_model_22.predict(x_test_series) 
    
    predictions_df["target_predict"] = np.argmax(predictions, axis=1)    
    
    mapping = {'O-NR':0, 'P-NR':1, 'Neutral':2, 'Weak':3, 'Strong':4}
    mapping_reverse = {0: 'O-NR', 1: 'P-NR', 2: 'Neutral', 3: 'Weak', 4: 'Strong'}
    label_to_score = {'O-NR': -2, 'P-NR': -1, 'Neutral': 0, 'Weak': 0.5, 'Strong': 1}
    
    predictions_df['label'] = predictions_df["target_predict"].map(mapping_reverse)
    predictions_df['label_score'] = predictions_df["label"].map(label_to_score)
    
    filename = file_.replace("automated_labelling/","")

    full_filename = os.path.join("labelled", filename)
    print(full_filename)

    print("\n\n\n")
    print(predictions_df)
    
    predictions_df.to_csv(full_filename,index=False)
    print(full_filename + ' saved')

