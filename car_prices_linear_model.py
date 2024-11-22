import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos
file_path = "Car_Prices_Poland_Cleaned.csv"
data = pd.read_csv(file_path)

data = data.drop(columns=['id'])

# Seleccionar solo las columnas numéricas para la matriz de correlación
numeric_data = data.select_dtypes(include=["float64", "int64"])

# Heatmap de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matriz de Correlación")
plt.show()

# Variables independientes y dependientes
X = data[["mileage", "vol_engine", "car_age"]]  # Seleccionar las columnas relevantes
y = data["price"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Visualización de predicciones vs valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.title("Valores Reales vs Predicciones")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.show()
