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
        
        # On récupère le chemin absolu du CSV situé dans le même dossier
        csv_path = os.path.join(os.path.dirname(__file__), 'cluster1data5120w512.csv')
        df = pd.read_csv(csv_path)

        # On sépare les caractéristiques (X) de la cible (y)
        # Supposons qu'on veut prédire 'Cluster' à partir de 'CO2' et 'T(°C)'
        X = df[['CO2', 'T(°C)']].values
        y = df['Cluster'].values

        # Transformation en classification binaire :
        # On prend la classe 1 comme positive, le reste comme négatif
        y_binary = np.where(y == 1, 1., -1.)

        return dict(X=X, y=y_binary)
