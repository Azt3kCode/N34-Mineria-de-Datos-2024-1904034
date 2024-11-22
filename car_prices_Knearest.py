import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
file_path = "Car_Prices_Poland_Cleaned.csv"
data = pd.read_csv(file_path)

# 1. Preprocesamiento
data = data.drop(columns=["id"])

# Convertir las variables categóricas a números
label_encoders = {}
categorical_cols = data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Seleccionar las características (X) y la variable objetivo (y)
X = data.drop(columns=["fuel"])  # Cambia 'fuel' por la columna objetivo deseada
y = data["fuel"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Entrenamiento del modelo KNN
k = 5  # Número de vecinos
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 3. Predicción y evaluación
y_pred = knn.predict(X_test)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoders["fuel"].classes_, yticklabels=label_encoders["fuel"].classes_)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Verdad")
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=label_encoders["fuel"].classes_))

# Precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo KNN (k={k}): {accuracy:.2f}")
