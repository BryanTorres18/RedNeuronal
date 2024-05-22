import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.svm import SVR

# Cargar el dataset
data = pd.read_csv('Dataset/housing.csv')

# Manejar valores faltantes
data.dropna(inplace=True)

features = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                 'population', 'households', 'median_income']]
target = data['median_house_value']

# Reducir el tamaño del conjunto de datos para una muestra inicial
_, features_sample, _, target_sample = train_test_split(features, target, test_size=0.1, random_state=42)

# Normalizar características
scaler = StandardScaler()
features_sample = scaler.fit_transform(features_sample)

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(features_sample, target_sample, test_size=0.2, random_state=42)

# Optimización de parámetros SVM
param_grid = {'C': [1, 10, 100, 1000],
              'gamma': [0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
svm = SVR()
svm_grid = GridSearchCV(svm, param_grid, cv=5, verbose=1, n_jobs=-1)
svm_grid.fit(X_train, y_train)

print("Mejores parámetros encontrados para SVM:")
print(svm_grid.best_params_)

# Mejor modelo SVM
best_svm = svm_grid.best_estimator_

# Entrenar el modelo SVM
best_svm.fit(X_train, y_train)

# Realizar predicciones en el conjunto de validación
y_pred_svm = best_svm.predict(X_val)

# Métricas de evaluación para SVM
mse_svm = np.mean((y_pred_svm - y_val) ** 2)
print(f'\nMean Squared Error (MSE) para SVM: {mse_svm:.2f}')

# Binarizar las predicciones para la matriz de confusión y F1 Score
threshold = np.median(data['median_house_value'])
y_val_bin = (y_val > threshold).astype(int)
y_pred_bin_svm = (y_pred_svm > threshold).astype(int)

# Manejo de casos donde no hay muestras predichas
if np.sum(y_pred_bin_svm) == 0:
    precision_svm = 0.0
else:
    precision_svm = precision_score(y_val_bin, y_pred_bin_svm)

# Calcular F1 Score, accuracy y precisión para SVM
f1_svm = f1_score(y_val_bin, y_pred_bin_svm)
accuracy_svm = accuracy_score(y_val_bin, y_pred_bin_svm)

print(f'F1 Score para SVM: {f1_svm:.2%}')
print(f'Exactitud para SVM: {accuracy_svm:.2%}')
print(f'Precisión para SVM: {precision_svm:.2%}')

# Matriz de confusión para SVM
conf_matrix_svm = confusion_matrix(y_val_bin, y_pred_bin_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Actual')
plt.title('Matriz de Confusión para SVM')
plt.show()




