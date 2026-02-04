"""
Preprocessing utilities for 
condition based crop recommendation system
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from pathlib import Path

from src.config import CLEANED_DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN


# LOAD PRE-CLEANED DATA

def load_data(data_path: Path = CLEANED_DATA_PATH):

    """
    This will pre-load the data.
    - Removing leading or lagging white spaces from column name
    - Return Clean data
    """

    df = pd.read_csv(data_path)

    # Normalizing Column names

    df.columns = [str(c).strip() for c in df.columns.to_list()]

    return df



# BUILDING PREPROCESSOR

def preprocessor(df_or_X: pd.DataFrame):

    #creating a backup

    df = df_or_X.copy()

    # if target column is present drop that column

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns = TARGET_COLUMN)
    
    else:

        df = df

    # Numeric Pipeline: impute --> Scale

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler())
    ])

    return pipeline


# TRAIN - TEST SPLIT

def split_data(df: pd.DataFrame):

    df = df.copy()

    # if target column is missing

    if TARGET_COLUMN not in df.columns:
        raise KeyError (f"TARGET COLUMN {TARGET_COLUMN} is not present in the DataFrame column: {df.columns.to_list()}")
    
    X = df.drop(columns= [TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= TEST_SIZE, random_state= RANDOM_STATE)

    return X_train, X_test, y_train, y_test


print ("Preprocessing is executed")