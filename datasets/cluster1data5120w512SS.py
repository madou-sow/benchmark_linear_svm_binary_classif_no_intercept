from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):

    name = "svm_cluster_ss"

    def get_data(self):
        # Récupération du chemin du fichier CSV
        csv_path = os.path.join(
            os.path.dirname(__file__), 'cluster1data5120w512.csv'
        )
        df = pd.read_csv(csv_path)

        # Sélection des caractéristiques (X) et de la cible (y)
        X = df[['CO2', 'T(°C)']].values
        y = df['Cluster'].values

        # Transformation en classification binaire :
        # On prend la classe 1 comme positive (1), le reste comme négative (-1)
        y_binary = np.where(y == 1, 1., -1.)

        # --- ETAPE DE CENTRAGE-REDUCTION (SCALING) ---
        # On transforme les données pour qu'elles aient une moyenne de 0 
        # et un écart-type de 1. Cela améliore drastiquement le conditionnement
        # de la matrice et la vitesse de convergence des solveurs.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return dict(X=X_scaled, y=y_binary)
