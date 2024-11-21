import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
file_path = "Car_Prices_Poland_Cleaned.csv"
data = pd.read_csv(file_path)

# Asegurarse de que no se incluya la columna 'id' en las pruebas
data = data.drop(columns=['id'])

# 1. Prueba ANOVA: Diferencia de precios por tipo de combustible
# Comprobamos si hay diferencias significativas en los precios de autos según el tipo de combustible
anova_data = [data[data['fuel'] == fuel]['price'] for fuel in data['fuel'].unique()]
anova_result = stats.f_oneway(*anova_data)

print("\nResultado de la prueba ANOVA para precios por tipo de combustible:")
if anova_result.pvalue < 0.05:
    print(f"Hay diferencias significativas entre los tipos de combustible. p-value = {anova_result.pvalue}")
else:
    print(f"No hay diferencias significativas entre los tipos de combustible. p-value = {anova_result.pvalue}")

# 2. T-test: Comparación de precios entre dos marcas (ejemplo: Audi vs Opel)
audi_prices = data[data['mark'] == 'audi']['price']
opel_prices = data[data['mark'] == 'opel']['price']
t_test_result = stats.ttest_ind(audi_prices, opel_prices)

print("\nResultado de la prueba T para comparar Audi y Opel:")
if t_test_result.pvalue < 0.05:
    print(f"Las medias de precios entre Audi y Opel son significativamente diferentes. p-value = {t_test_result.pvalue}")
else:
    print(f"No hay diferencias significativas en las medias de precios entre Audi y Opel. p-value = {t_test_result.pvalue}")

# 3. Kruskal-Wallis Test: Comparación de precios entre más de dos marcas
# Usamos este test cuando los datos no siguen una distribución normal
kruskal_data = [data[data['mark'] == mark]['price'] for mark in data['mark'].unique() if len(data[data['mark'] == mark]) > 30]
kruskal_result = stats.kruskal(*kruskal_data)

print("\nResultado de la prueba Kruskal-Wallis para comparar precios entre marcas:")
if kruskal_result.pvalue < 0.05:
    print(f"Las medias de precios entre marcas son significativamente diferentes. p-value = {kruskal_result.pvalue}")
else:
    print(f"No hay diferencias significativas en las medias de precios entre marcas. p-value = {kruskal_result.pvalue}")

# Visualización de la distribución de precios para las marcas seleccionadas
plt.figure(figsize=(12, 6))
sns.boxplot(x='mark', y='price', data=data, palette='Set3')
plt.title('Distribución de Precios por Marca')
plt.xlabel('Marca')
plt.ylabel('Precio')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()