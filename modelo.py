import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

data = pd.read_csv('Dataset/housing.csv')

data.dropna(inplace=True)

# Convertir la columna 'ocean_proximity' a valores numéricos
label_encoder = LabelEncoder()
data['ocean_proximity'] = label_encoder.fit_transform(data['ocean_proximity'])

features = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                 'population', 'households', 'median_income', 'ocean_proximity']]
target = data['median_house_value']

scaler = StandardScaler()
features = scaler.fit_transform(features)

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

y_pred = model.predict(X_val)

# Función para predecir el precio de la vivienda basado en características específicas
def predict_price(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]])
    input_data = scaler.transform(input_data)
    predicted_price = model.predict(input_data)
    return predicted_price[0][0]


def show_prediction():
    try:
        total_rooms = float(entry_total_rooms.get())
        total_bedrooms = float(entry_total_bedrooms.get())
        households = float(entry_households.get())
        median_income = float(entry_median_income.get())
        ocean_proximity = combo_ocean_proximity.get()

        if total_rooms <= 0 or total_bedrooms <= 0 or households <= 0 or median_income <= 0:
            raise ValueError("Los valores deben ser mayores que cero.")

        # Convertir la selección de 'ocean_proximity' a un valor numérico
        ocean_proximity_num = label_encoder.transform([ocean_proximity])[0]

        longitude = data['longitude'].mean()
        latitude = data['latitude'].mean()
        housing_median_age = data['housing_median_age'].mean()
        population = data['population'].mean()

        predicted_price = predict_price(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity_num)
        messagebox.showinfo("Precio Predicho de la Vivienda", f"Precio Predicho de la Vivienda: ${predicted_price:.2f}")
    except ValueError as e:
        messagebox.showerror("Entrada no válida", str(e))

root = tk.Tk()
root.title("Predicción de Precios de Viviendas en California")

style = ttk.Style()
style.configure('TLabel', font=('Arial', 12))
style.configure('TButton', font=('Arial', 12), padding=10)
style.configure('TEntry', font=('Arial', 12))

label_total_rooms = ttk.Label(root, text="Ingrese el número total de habitaciones (por bloque):")
label_total_rooms.pack(pady=5)

entry_total_rooms = ttk.Entry(root)
entry_total_rooms.pack(pady=5)

label_total_bedrooms = ttk.Label(root, text="Ingrese el número total de dormitorios (por bloque):")
label_total_bedrooms.pack(pady=5)

entry_total_bedrooms = ttk.Entry(root)
entry_total_bedrooms.pack(pady=5)

label_households = ttk.Label(root, text="Ingrese el número de hogares (por bloque):")
label_households.pack(pady=5)

entry_households = ttk.Entry(root)
entry_households.pack(pady=5)

label_median_income = ttk.Label(root, text="Ingrese el ingreso medio (en decenas de miles):")
label_median_income.pack(pady=5)

entry_median_income = ttk.Entry(root)
entry_median_income.pack(pady=5)

label_ocean_proximity = ttk.Label(root, text="Seleccione la proximidad al océano:")
label_ocean_proximity.pack(pady=5)

combo_ocean_proximity = ttk.Combobox(root, values=["NEAR BAY", "INLAND", "<1H OCEAN", "ISLAND"], state="readonly")
combo_ocean_proximity.pack(pady=5)
combo_ocean_proximity.current(0)

button_predict = ttk.Button(root, text="Predecir Precio de la Vivienda", command=show_prediction)
button_predict.pack(pady=20)

root.mainloop()





