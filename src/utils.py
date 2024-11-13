import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_train_test_data():
    '''
    Returns a list of datasets to use for each participant for each visit. It outputs all of
    these datasets as a dictionary. For the training data, the label is the time_to_reach. For
    testing, we are interested in the averaged value for the gt_difficulty. 
    '''
    stroke_data = pd.read_csv('../simplified_data/poststroke_data.csv')
    neurotypical_data = pd.read_csv('../simplified_data/neurotypical_data.csv')

    datasets = []

    #seperate each data by individual participant
    for pid in stroke_data['pid'].unique():
        pid_data = stroke_data.query(f'pid == {pid}').copy()

        #further seperate by visit
        for visit in stroke_data['visit'].unique():
            visit_data = pid_data.query(f'visit == "{visit}"').copy()

            #ignore the visit if it has less than 10 data points
            if len(visit_data) < 10:
                continue

            side = visit_data['side'].iloc[0]
            
            X = visit_data[['x','y','z']].values
            y = visit_data[['time_to_press', 'gt_difficulty']].values

            strokeX_train, strokeX_test, strokey_train, strokey_test = train_test_split(X, y, test_size=0.2, random_state=42)

            #get the neurotypical data for this side
            neurotypical_slice = neurotypical_data.query(f'side == "{side}"').copy()
            X = neurotypical_slice[['x','y','z']].values
            y = neurotypical_slice[['time_to_press']].values

            neuroX_train, neuroy_train = X, y

            datasets.append({
                'stroke_X_train': strokeX_train,
                'stroke_X_test': strokeX_test,
                'stroke_y_train': strokey_train[:,0].flatten(),
                'stroke_y_test': strokey_test[:,1].flatten(),
                'neuro_X_train': neuroX_train,
                'neuro_y_train': neuroy_train.flatten(),
            })

    return datasets

