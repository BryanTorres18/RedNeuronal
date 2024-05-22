import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

# Cargar el dataset
data = pd.read_csv('Dataset/housing.csv')

# Manejar valores faltantes
data.dropna(inplace=True)

features = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                 'population', 'households', 'median_income']]
target = data['median_house_value']

# Normalizamos las características
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Dividivimos los datos en entrenamiento y validación 80/20 (Entrenamiento/Validación): Usamos esta proporcon ya que
# comúnmente utilizada que proporciona un buen equilibrio entre tener suficientes datos para entrenar y validar el
# modelo. Es util ya que nuestro dataset es razonablemente grande.
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Creacion del modelo de red neuronal
model = Sequential([
    Dense(128, activation='relu', input_shape=(features.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Entrenamos el modelo
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Grafico del porcentaje de entrenamiento y validación
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida de Entrenamiento y Validación')
plt.show()

# Realiza las predicciones en el conjunto de validación
y_pred = model.predict(X_val)

# Binarizar las predicciones para la matriz de confusión y F1 Score
threshold = np.median(data['median_house_value'])
y_val_bin = (y_val > threshold).astype(int)
y_pred_bin = (y_pred.flatten() > threshold).astype(int)

# Matriz de confusión
conf_matrix = confusion_matrix(y_val_bin, y_pred_bin)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()

# F1 Score, accuracy y precisión
f1 = f1_score(y_val_bin, y_pred_bin)
accuracy = accuracy_score(y_val_bin, y_pred_bin)
precision = precision_score(y_val_bin, y_pred_bin)

print(f'F1 (RNN): {f1:.2%}')
print(f'Exactitud (RNN): {accuracy:.2%}')
print(f'Precisión (RNN): {precision:.2%}')










