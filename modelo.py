import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# Cargar el dataset
data = pd.read_csv('Dataset/housing.csv')

# Manejar valores faltantes
data.dropna(inplace=True)

# Seleccionar características y objetivo
features = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                 'population', 'households', 'median_income']]
target = data['median_house_value']

# Normalizar características
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = Sequential([
    Dense(128, activation='relu', input_shape=(features.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo con early stopping
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Realizar predicciones en el conjunto de validación
y_pred = model.predict(X_val)

# Función para predecir el precio de la vivienda basado en características específicas
def predict_price(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income):
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])
    input_data = scaler.transform(input_data)
    predicted_price = model.predict(input_data)
    return predicted_price[0][0]

# Función para crear el mapa de calor
def create_heatmap():
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x=data['longitude'], y=data['latitude'], hue=data['median_house_value'], palette='coolwarm', s=100)
    plt.colorbar(scatter.collections[0], label='Median House Value')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Mapa de Calor de Precios de Viviendas en California')
    plt.show()

def show_prediction():
    try:
        total_rooms = float(entry_total_rooms.get())
        total_bedrooms = float(entry_total_bedrooms.get())
        households = float(entry_households.get())
        median_income = float(entry_median_income.get())

        # Usar valores promedio para las otras características
        longitude = data['longitude'].mean()
        latitude = data['latitude'].mean()
        housing_median_age = data['housing_median_age'].mean()
        population = data['population'].mean()

        predicted_price = predict_price(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income)
        messagebox.showinfo("Precio Predicho de la Vivienda", f"Precio Predicho de la Vivienda: ${predicted_price:.2f}")
    except ValueError:
        messagebox.showerror("Entrada no válida", "Por favor, ingrese valores numéricos válidos para todos los campos.")

root = tk.Tk()
root.title("Predicción de Precios de Viviendas en California")

style = ttk.Style()
style.configure('TLabel', font=('Arial', 12))
style.configure('TButton', font=('Arial', 12), padding=10)
style.configure('TEntry', font=('Arial', 12))

# Crear los widgets
label_total_rooms = ttk.Label(root, text="Ingrese el número total de habitaciones:")
label_total_rooms.pack(pady=5)

entry_total_rooms = ttk.Entry(root)
entry_total_rooms.pack(pady=5)

label_total_bedrooms = ttk.Label(root, text="Ingrese el número total de dormitorios:")
label_total_bedrooms.pack(pady=5)

entry_total_bedrooms = ttk.Entry(root)
entry_total_bedrooms.pack(pady=5)

label_households = ttk.Label(root, text="Ingrese el número de hogares:")
label_households.pack(pady=5)

entry_households = ttk.Entry(root)
entry_households.pack(pady=5)

label_median_income = ttk.Label(root, text="Ingrese el ingreso medio (en decenas de miles):")
label_median_income.pack(pady=5)

entry_median_income = ttk.Entry(root)
entry_median_income.pack(pady=5)

button_predict = ttk.Button(root, text="Predecir Precio de la Vivienda", command=show_prediction)
button_predict.pack(pady=20)

button_heatmap = ttk.Button(root, text="Mostrar Mapa de Calor", command=create_heatmap)
button_heatmap.pack(pady=10)

root.mainloop()


