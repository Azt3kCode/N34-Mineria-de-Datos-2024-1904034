import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar datos
file_path = "Car_Prices_Poland_Cleaned.csv"
data = pd.read_csv(file_path)

# Simular datos de series temporales: promedio de precio por año
time_series_data = data.groupby('year')['price'].mean().reset_index()
time_series_data.rename(columns={'year': 'time', 'price': 'avg_price'}, inplace=True)

# Variables independientes (tiempo) y dependientes (precio promedio)
X = time_series_data['time'].values.reshape(-1, 1)
y = time_series_data['avg_price'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Mostrar coeficientes del modelo
print("\nCoeficiente (pendiente):", model.coef_[0])
print("Intersección:", model.intercept_)

# Predicción para nuevos datos (por ejemplo, años futuros)
future_years = np.array([2024, 2025, 2026]).reshape(-1, 1)
future_prices = model.predict(future_years)

print("\nPredicciones para años futuros:")
for year, price in zip(future_years.flatten(), future_prices):
    print(f"Año {year}: Precio promedio estimado = {price:.2f}")

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Datos reales", color="blue")
plt.plot(X, model.predict(X), label="Regresión lineal", color="red")
plt.scatter(future_years, future_prices, label="Predicción futura", color="green")
plt.title("Forecasting: Precio promedio de autos por año")
plt.xlabel("Año")
plt.ylabel("Precio promedio")
plt.legend()
plt.grid()
plt.show()
