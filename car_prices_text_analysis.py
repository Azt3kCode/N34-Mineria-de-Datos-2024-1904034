import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cargar datos
file_path = "Car_Prices_Poland_Cleaned.csv"
data = pd.read_csv(file_path)

# Concatenar las columnas de texto que puedan contener información útil
text_data = (
    data['mark'].astype(str) + " " +
    data['model'].astype(str) + " " +
    data['generation_name'].astype(str) + " " +
    data['fuel'].astype(str) + " " +
    data['city'].astype(str) + " " +
    data['province'].astype(str)
)

# Unir todos los datos de texto en una sola cadena
text = " ".join(text_data)

# Crear el WordCloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis',
    max_words=200
).generate(text)

# Mostrar el WordCloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud de los datos de texto", fontsize=16)
plt.tight_layout()
plt.show()
