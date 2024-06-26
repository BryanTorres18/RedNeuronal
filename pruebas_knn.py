import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv('Dataset/housing.csv')

data.dropna(inplace=True)

features = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                 'population', 'households', 'median_income']]
target = data['median_house_value']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f"Mejor número de vecinos encontrado: {grid_search.best_params_['n_neighbors']}")

knn_model = KNeighborsRegressor(n_neighbors=grid_search.best_params_['n_neighbors'])
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_val)

mse_knn = np.mean((y_pred_knn - y_val) ** 2)
print(f'Mean Squared Error (MSE) para KNN: {mse_knn:.2f}')

threshold = np.median(data['median_house_value'])
y_val_bin = (y_val > threshold).astype(int)
y_pred_bin_knn = (y_pred_knn > threshold).astype(int)
conf_matrix_knn = confusion_matrix(y_val_bin, y_pred_bin_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Actual')
plt.title('Matriz de Confusión para KNN')
plt.show()

f1_knn = f1_score(y_val_bin, y_pred_bin_knn)
accuracy_knn = accuracy_score(y_val_bin, y_pred_bin_knn)
precision_knn = precision_score(y_val_bin, y_pred_bin_knn)

print(f'F1 Score para KNN: {f1_knn:.2%}')
print(f'Exactitud para KNN: {accuracy_knn:.2%}')
print(f'Precisión para KNN: {precision_knn:.2%}')
