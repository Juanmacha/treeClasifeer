import pandas as pd
import numpy as np   # 👈 IMPORTANTE para manejar NaN
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# cargamos el archivo CSV (asegúrate que esté en el mismo directorio del script)
data = pd.read_csv('casas_sucias.csv')

# limpieza de datos
data = data.drop_duplicates()  # eliminar filas duplicadas
data = data.replace("?", np.nan)  # reemplazar "?" por NaN
data = data.dropna()  # eliminar filas con NaN

# --- LIMPIEZA DE COLUMNAS ---

# superficie: quitar "m2" y convertir a número
data['superficie'] = data['superficie'].str.replace('m2', '', regex=False).astype(float)

# habitaciones: quitar texto y mapear palabras a números si existen
data['habitaciones'] = (
    data['habitaciones']
    .str.replace('habitaciones', '', regex=False)
    .replace({
        'una': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
        'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10
    })
    .astype(int)
)

# antiguedad: quitar "años" y mapear palabras a números
data['antiguedad'] = (
    data['antiguedad']
    .str.replace('años', '', regex=False)
    .replace({
        'nueva': 0, 'un': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
        'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10
    })
    .astype(int)
)

# convertir categóricos a numéricos
data['ubicacion'] = data['ubicacion'].astype('category').cat.codes  

# binarizar precio (1 = caro, 0 = barato)
data['precio'] = data['precio'].apply(lambda x: 1 if x > 300000 else 0)

print(data.info())
print(data.head())

# separar características y etiquetas
X = data.drop('precio', axis=1)
y = data['precio']

# dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# crear y entrenar el modelo
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# hacer predicciones
y_pred = model.predict(X_test)

# evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# visualizar el árbol de decisión
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=['Barato', 'Caro'], filled=True)
plt.title('Árbol de Decisión para Clasificación de Precios de Casas')
plt.show()

# guardar el modelo entrenado
joblib.dump(model, 'decision_tree_model.pkl')
# guadar feactures
joblib.dump(X.columns.tolist(), 'model_features.pkl')
print('Modelo guardado como decision_tree_model.pkl')
