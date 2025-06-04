# Contenu du dataset
# Chaque ligne contient :

# Entrées (features) :

# roll, pitch, yaw (angles IMU)

# ax, ay, az (accélération)

# lat, lon, alt (position GPS)

# h1, h2, h3, h4 (valeurs actuelles des hélices)

# Sorties (targets) :

# delta_h1, delta_h2, delta_h3, delta_h4 (corrections à appliquer aux hélices)

# Tu veux entraîner un modèle de stabilisation de drone basé sur des données en temps réel (capteurs IMU, GPS, hélices). Puisque tu n'as pas encore de dataset mais tu as la possibilité de recueillir des données, voici un plan clair pour constituer un ensemble de données cohérent et exploitable pour l'entraînement d'un modèle :

# 1. Définir les entrées (features)
# Ce sont les données que tu peux lire des capteurs du drone :

# IMU (gyroscope) : roll, pitch, yaw (3 angles)

# Accéléromètre : accélérations ou vitesses selon les axes ax, ay, az

# GPS : latitude, longitude, altitude

# Vitesse des hélices : [h1, h2, h3, h4] valeurs entre 0.0 et 1.0


# 1. Import des librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 2. Chargement du dataset
df = pd.read_csv("drone_stabilization_dataset.csv")

# 3. Inspection rapide
print("Aperçu du dataset :")
print(df.head())
print("\nStatistiques descriptives :")
print(df.describe())

# 4. Nettoyage de base
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())
# (Optionnel) Suppression ou imputation si valeurs manquantes
# df = df.dropna()  # ou utiliser df.fillna()

# 5. Séparation features/targets
features = ['roll', 'pitch', 'yaw', 'ax', 'ay', 'az', 'lat', 'lon', 'alt', 'h1', 'h2', 'h3', 'h4']
targets = ['delta_h1', 'delta_h2', 'delta_h3', 'delta_h4']
X = df[features]
y = df[targets]

# 6. Normalisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Entraînement du modèle
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# 9. Évaluation
y_pred = model.predict(X_test)
print("Erreur quadratique moyenne :", mean_squared_error(y_test, y_pred))
