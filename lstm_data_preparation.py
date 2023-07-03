import numpy as np
import pandas as pd
def lstm_data(df,t,w):

    y = df
    x = df #.drop(columns=[t])

    y1 = y.iloc[w:].values # remove first w values and transform to np array

    lx = []
    for i in range(len(x)-w):
        xi = x.iloc[i:i+w] # get all 12-timestep time series
        lx.append(xi)

    x_final = np.asarray(lx) # np array
    return x_final, y1
