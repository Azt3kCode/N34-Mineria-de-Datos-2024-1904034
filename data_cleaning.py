import pandas as pd

# Cargar datos
data = pd.read_csv("Car_Prices_Poland_Kaggle.csv")

# 1. Valores nulos
data['generation_name'] = data['generation_name'].fillna("Desconocido")
data = data.dropna()

# 2. Eliminar duplicados
data = data.drop_duplicates()

# 3. Eliminar valores extremos
data = data[(data['price'] > 500) & (data['mileage'] > 0) & (data['vol_engine'] > 0)]

# 4. Estandarizar texto
data['fuel'] = data['fuel'].str.lower()
fuel_map = {'gasoline': 'petrol', 'diesel': 'diesel', 'cng': 'cng', 'hybrid': 'hybrid'}
data['fuel'] = data['fuel'].map(fuel_map).fillna(data['fuel'])

# 5. Convertir tipos de datos
data['year'] = data['year'].astype(int)
data['price'] = data['price'].astype(float)

# 6. Crear columna de edad del auto
import datetime
current_year = datetime.datetime.now().year
data['car_age'] = current_year - data['year']

# Verificar dataset limpio
print(data.head())

# Archivo limpio
data.to_csv("Car_Prices_Poland_Cleaned.csv", index=False)
