from utils import get_train_test_data
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor



datasets = get_train_test_data()
eval_datasets = datasets[:len(datasets)//2]
test_datasets = datasets[len(datasets)//2:]

for dataset in eval_datasets:

    # Train the stroke model
    stroke_model = RandomForestRegressor()
    stroke_model.fit(dataset['stroke_X_train'], dataset['stroke_y_train'])

    # Train the neurotypical model
    neurotypical_model = RandomForestRegressor()
    neurotypical_model.fit(dataset['neuro_X_train'], dataset['neuro_y_train'])

    # Test the models
    preds = stroke_model.predict(dataset['stroke_X_test']) - neurotypical_model.predict(dataset['stroke_X_test'])
    
    print(np.abs(preds - dataset['stroke_y_test']))
