import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
file_path = "Car_Prices_Poland_Cleaned.csv"
data = pd.read_csv(file_path)

# Selección de columnas relevantes para clustering
columns_to_cluster = ['price', 'mileage', 'car_age', 'vol_engine']
data_cluster = data[columns_to_cluster]

# Escalar los datos (importante para K-Means)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cluster)

# Encontrar el número óptimo de clústeres usando la "Elbow Method"
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Gráfico para la Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Método del Codo para determinar k')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Inercia')
plt.grid()
plt.show()

# Elegir un valor de k (por ejemplo, 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Análisis de los clústeres
print("\nCentroides de los Clústeres:")
print(kmeans.cluster_centers_)

# Visualización: Clústeres en dos dimensiones (por ejemplo, 'price' y 'mileage')
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['price'], y=data['mileage'], hue=data['cluster'], palette='Set2', s=50)
plt.title('Clústeres según Precio y Kilometraje')
plt.xlabel('Precio')
plt.ylabel('Kilometraje')
plt.legend(title='Clúster')
plt.tight_layout()
plt.show()

# Visualización: Boxplot por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster', y='price', data=data, palette='Set3')
plt.title('Distribución de Precio por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Precio')
plt.tight_layout()
plt.show()
