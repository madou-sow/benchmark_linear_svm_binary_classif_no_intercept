'''
Implemented : Mamadou S.
Date : 2026, April 3
Definition: A Python script that inherits from BaseDataset.
Role: Loads the CSV file above, cleans the data, and transforms it into a binary classification problem
     (e.g., Class 1 vs. the rest) to ensure compatibility with the SVM objective.

cluster1data5120w512.csv
Definition: A raw data file in CSV format.
Role: Contains actual measurements (CO2, Temperature) associated with clusters. 
It is the "field" data source.
'''

from benchopt import BaseDataset, safe_import_context
with safe_import_context() as import_ctx:
 import numpy as np
 import pandas as pd
 import os


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "svm_cluster"

    def get_data(self):
        
        # We retrieve the absolute path of the CSV located in the same folder
        csv_path = os.path.join(os.path.dirname(__file__), 'cluster1data5120w512.csv')
        df = pd.read_csv(csv_path)

        # We separate the characteristics (X) from the target (y)
        # Suppose we want to predict 'Cluster' from 'CO2' and 'T(°C)'
        X = df[['CO2', 'T(°C)']].values
        y = df['Cluster'].values

        # Transformation into binary classification:
        # We take class 1 as positive, the rest as negative
        y_binary = np.where(y == 1, 1., -1.)

        return dict(X=X, y=y_binary)
