'''
Implemented : Mamadou S.
Date : 2026, April 3
'''

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):

    name = "svm_cluster_ss"

    def get_data(self):
        # Retrieving the CSV file path
        csv_path = os.path.join(
            os.path.dirname(__file__), 'cluster1data5120w512.csv'
        )
        df = pd.read_csv(csv_path)

        # Feature (X) and target (y) selection
        X = df[['CO2', 'T(°C)']].values
        y = df['Cluster'].values

        # Transformation into binary classification:
        # We take class 1 as positive (1), the rest as negative (-1)
        y_binary = np.where(y == 1, 1., -1.)

        # --- CENTERING-SCALING STAGE ---
        # We transform the data so that it has a mean of 0 
        # and a standard deviation of 1. This drastically improves the conditioning 
        # of the matrix and the convergence speed of the solvers.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return dict(X=X_scaled, y=y_binary)
