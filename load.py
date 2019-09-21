
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load():
    df = pd.read_csv('./Datasets/uci-ionosphere/ionosphere_data_kaggle.csv')
    df.head()
    X = df.drop('label',axis=1).values
    Y = df['label'].values
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,stratify=Y)
    return [X_train,X_test,Y_train,Y_test]