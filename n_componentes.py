from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datos import baja_datos, tickers
import matplotlib.pyplot as plt
import numpy as np

df = baja_datos(tickers).pct_change(1).dropna(axis=0)
print (df)

# Normalización
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Aplica PCA
pca = PCA()
pca.fit(scaled_data)

# Varianza explicada por cada componente
explained_variance = pca.explained_variance_ratio_

# Gráfico de Scree
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada')
plt.grid(True)
plt.show()

# Acumulación de varianza explicada
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8,6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Varianza Explicada Acumulada')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.grid(True)
plt.show()

# Imprime la varianza explicada acumulada
print('Varianza explicada acumulada:', cumulative_variance)

# Número de componentes con valores propios mayores que 1
eigenvalues = pca.explained_variance_
num_components = sum(eigenvalue > 1 for eigenvalue in eigenvalues)
print('Número de componentes con eigenvalues > 1:', num_components)

