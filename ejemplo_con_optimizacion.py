# Esto es una optimización dummy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datos import baja_datos, tickers

# Baja los datos y calcula los retornos diarios
df = baja_datos(tickers).pct_change(1).dropna(axis=0)

# Normalización
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Aplica PCA
pca = PCA(n_components=4)  # Cambia el número de componentes según sea necesario
principal_components = pca.fit_transform(scaled_data)

# Crea un DataFrame con los componentes principales
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4'])

# Cargas (loadings)
loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df.columns, columns=['PC1', 'PC2', 'PC3', 'PC4'])

print("Cargas (Loadings):")
print(loadings_df)

# Función de optimización (ejemplo: maximizar la suma de PC1 y PC2)
def objective(x):
    return -(x[0] + x[1])  # Negativo para maximizar en lugar de minimizar

# Restricciones (ejemplo: las variables deben estar en un rango específico)
constraints = (
    {'type': 'ineq', 'fun': lambda x: x[0] + 1},  # PC1 >= -1
    {'type': 'ineq', 'fun': lambda x: x[1] + 1},  # PC2 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[0]},  # PC1 <= 1
    {'type': 'ineq', 'fun': lambda x: 1 - x[1]},  # PC2 <= 1
)

# Valores iniciales
x0 = [0, 0, 0, 0]

# Optimización
result = minimize(objective, x0, constraints=constraints)
optimized_pcs = result.x

print("\nComponentes Principales Optimizados:")
print(optimized_pcs)

# Convertir los resultados optimizados al espacio original
optimized_original = np.dot(optimized_pcs, pca.components_) * scaler.scale_ + scaler.mean_

# Crear un DataFrame para mostrar los resultados en términos originales
optimized_original_df = pd.DataFrame([optimized_original], columns=df.columns)

print("\nResultados Optimizados en Términos Originales:")
print(optimized_original_df)