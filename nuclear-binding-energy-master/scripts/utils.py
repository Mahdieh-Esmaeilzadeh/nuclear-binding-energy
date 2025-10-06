# -*- coding: utf-8 -*-
# Script: utils.py
# Generated automatically from BindingEnergy.ipynb

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted

import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

# Reproducibility
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# Paths
DATA_PATH = "/content/data.csv"
ARTIFACT_DIR = "./artifacts"
CKPT_DIR = os.path.join(ARTIFACT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET_COLUMN = "binding_energy"

