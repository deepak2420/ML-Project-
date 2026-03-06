import numpy as np
import pandas as pd


def clean_data(data):

    # remove spaces in column names
    data.columns = data.columns.str.strip()

    # replace infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # remove missing rows
    data.dropna(inplace=True)

    return data


def convert_labels(data):

    data['Label'] = data['Label'].apply(
        lambda x: 0 if x == "BENIGN" else 1
    )

    return data


def sample_dataset(data, fraction=0.2):

    """
    Random sampling
    Example:
    if dataset = 1,000,000 rows
    fraction=0.2 → 200,000 rows
    """

    sampled_data = data.sample(frac=fraction, random_state=42)

    print("Sampled Dataset Shape:", sampled_data.shape)

    return sampled_data


def split_features(data):

    X = data.drop("Label", axis=1)
    y = data["Label"]

    return X, y