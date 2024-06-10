import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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

# Imprime la varianza explicada por cada componente
print('Varianza explicada por cada componente:', pca.explained_variance_ratio_)

# Cargas (loadings)
loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df.columns, columns=['PC1', 'PC2', 'PC3', 'PC4'])

print("\nCargas (Loadings):")
print(loadings_df)

# Visualización de las cargas
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='varianza individual')
plt.step(range(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', label='varianza acumulada')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada')
plt.legend(loc='best')
plt.grid(True)
plt.title('Varianza Explicada por los Componentes Principales')
plt.show()