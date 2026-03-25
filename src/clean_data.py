import pandas as pd

def normalize(train, test):
    cols = train.columns.drop('label')
    train[cols] = train[cols].astype(float)
    test[test.columns] = test[test.columns].astype(float)
    
    train.iloc[:, 1:] /= 255
    test /= 255
    return train, test