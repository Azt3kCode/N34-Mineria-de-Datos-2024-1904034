import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
file_path = "Car_Prices_Poland_Cleaned.csv"

# Cargar datos
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"El archivo '{file_path}' no fue encontrado. Verifica la ruta y el nombre del archivo.")
    exit()


data = data.drop(columns=['id'])

# Verificar las primeras filas
print("Primeras filas del dataset:")
print(data.head())

# Listado de columnas numéricas y categóricas
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# 1. Gráficos de Pie (Categóricos)
for col in categorical_columns[:2]:  # Limitar a las primeras 2 columnas categóricas
    plt.figure(figsize=(8, 6))
    data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, cmap='tab20')
    plt.title(f'Distribución de {col}')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

# 2. Histogramas (Numéricos)
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], bins=30, kde=True, color='blue')
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

# 3. Diagramas de Caja (Boxplots)
for col in numeric_columns[:3]:  # Limitar a las primeras 3 columnas numéricas
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, y=col, color='green')
    plt.title(f'Diagrama de Caja de {col}')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# 4. Gráficos de Dispersión (Scatter Plots)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='mileage', y='price', hue='fuel', palette='viridis', alpha=0.7)
plt.title('Dispersión entre Mileage y Price')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.legend(title='Fuel', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 5. Gráficos de Barras (Categóricos y Numéricos Combinados)
plt.figure(figsize=(10, 6))
top_brands = data['mark'].value_counts().head(10).index  # Top 10 marcas
sns.barplot(x='mark', y='price', data=data[data['mark'].isin(top_brands)], estimator='mean', ci=None, palette='muted')
plt.title('Precio Promedio por Marca (Top 10)')
plt.xlabel('Marca')
plt.ylabel('Precio Promedio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Gráfico de Correlación (Heatmap)
plt.figure(figsize=(10, 8))
corr_matrix = data[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.show()
