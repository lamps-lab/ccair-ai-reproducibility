# -*- coding: utf-8 -*-
"""
R_001_Research_Project__
|
R_001_binary_model_predictions.py
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


model_path = 'lemos_R_001_binary_label_predictor_009_12_ep.keras'
reloaded_model_22 = keras.models.load_model(model_path)

print("Model loaded")

zero_citation_list = ["RS_007","RS_070","RS_146"]
label_file_list = [i for i in glob.glob("automated_labelling/*.csv") if i.replace("automated_labelling/","")[:6] not in zero_citation_list]

print(len(label_file_list))
df_test_1 = pd.read_csv(label_file_list[12])

for file_ in label_file_list:
    df_ = pd.read_csv(file_)
    print(len(df_))

    predictions_df = df_.copy()

    predictions_df['target_predict'] = 0
    x_test_series = predictions_df['text']
        
    predictions = reloaded_model_22.predict(x_test_series)
    predictions_df["target_predict"] = np.argmax(predictions, axis=1)
    
    label_to_score = {'related':1,"not_related":0}
    mapping_reverse = {0: 'not_related', 1: 'related'}
    
    
    predictions_df['target_predict_label'] = predictions_df["target_predict"].map(mapping_reverse)
    # print("\n\n\n")
          
    filename = file_.replace("automated_labelling/","")

    full_filename = os.path.join("related_binary_labelled", filename)
    print(full_filename)

    print("\n\n\n")
    print(predictions_df)
    
    predictions_df.to_csv(full_filename,index=False)
    print(full_filename + ' saved')
