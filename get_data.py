
import pandas as pd
import numpy as np
def get_data():
    path = pd.read_excel(io='filepath.xlsx', header=None)
    path = list(np.ravel(path.values.tolist()))
    raw = np.load(path[0])
    epochs = raw['x']
    events = raw['y']

    for i in range(len(path) - 1):
        raw = np.load(path[i+1])
        epochs = np.concatenate((epochs, raw['x']))
        events = np.concatenate((events, raw['y']))
    return epochs, events


