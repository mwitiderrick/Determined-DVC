import tensorflow

import pandas as pd
import numpy as np
def load_training_data():
    df = pd.read_csv("data/mnist_train.csv")
    x_train = df.drop("label",axis=1)
    x_train = x_train.values
    y_train = df["label"].values
    return x_train, y_train


def load_validation_data():
    df = pd.read_csv("data/mnist_test.csv")
    x_test = df.drop("label",axis=1)
    x_test =  x_test.values
    y_test = df["label"].values
    return x_test, y_test

