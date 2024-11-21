import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
file_path = "Car_Prices_Poland_Cleaned.csv"

# Verificar las primeras filas del dataset
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"El archivo '{file_path}' no fue encontrado. Verifica la ruta y el nombre del archivo.")
    exit()

# 1. Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(data.describe())  # Numéricas
print("\nEstadísticas de columnas categóricas:")
print(data.describe(include=['object']))  # Categóricas

# 2. Agrupación de datos
# Agrupamos por 'mark' (Marca)
grouped_data = data.groupby('mark')['price'].agg(['mean', 'median', 'std', 'count']).reset_index()
print("\nEstadísticas agrupadas por Marca:")
print(grouped_data)

# Top 5 marcas con mayor precio promedio
top_5_brands = grouped_data.sort_values(by='mean', ascending=False).head(5)

# 3. Visualización
# Gráfico de barras: Precio promedio por marca (Top 5)
plt.figure(figsize=(10, 6))
sns.barplot(x='mark', y='mean', data=top_5_brands, palette='coolwarm')
plt.title('Precio Promedio por Marca (Top 5)')
plt.xlabel('Marca')
plt.ylabel('Precio Promedio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico adicional: Distribución de precios por tipo de combustible
plt.figure(figsize=(10, 6))
sns.boxplot(x='fuel', y='price', data=data, palette='Set2')
plt.title('Distribución de Precios por Tipo de Combustible')
plt.xlabel('Tipo de Combustible')
plt.ylabel('Precio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

