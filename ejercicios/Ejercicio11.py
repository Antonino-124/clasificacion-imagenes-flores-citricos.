from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# --- FUNCIONES DE EXTRACCIÓN (Necesarias para que el código funcione) ---

def hu_moments(img):
    # Calcula los 7 momentos de Hu (forma)
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    # Transformación logarítmica para estabilizar los valores
    return -np.sign(hu) * np.log10(np.abs(hu))

def haralick_features(img):
    # Características de textura simples (Media y Desviación)
    # Nota: El Haralick real requiere la librería 'mahotas'
    return np.array([img.mean(), img.std()])

def color_histogram(img):
    # Histograma de color en espacio HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# --- CONFIGURACIÓN DE RUTA ---
# Usamos 'r' para que Windows reconozca las barras correctamente
dataset_path = str(Path(__file__).resolve().parents[1] / "dataset" / "dataset")

x = []
y = []

# --- CARGA Y PROCESAMIENTO ---
print("Cargando imágenes...")
if not os.path.exists(dataset_path):
    print(f"ERROR: La ruta no existe: {dataset_path}")
else:
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Extraer características
                    hu = hu_moments(gray)
                    har = haralick_features(gray)
                    color = color_histogram(img)
                    
                    # Unir todas las características en un solo vector
                    features = np.hstack([hu, har, color])
                    x.append(features)
                    y.append(folder) # La etiqueta es el nombre de la carpeta

    # Convertir a arreglos de numpy
    x = np.array(x)
    y = np.array(y)

    if x.shape[0] > 0:
        print(f"Datos cargados: {x.shape[0]} imágenes encontradas.")
        
        # SEPARAR DATOS
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # ENTRENAR CLASIFICADOR
        clf = SVC(kernel="rbf", gamma='auto')
        clf.fit(x_train, y_train)

        # EVALUAR EN CONSOLA
        y_pred = clf.predict(x_test)
        print("\n--- REPORTE DE CLASIFICACIÓN ---")
        print(classification_report(y_test, y_pred))

        # --- MOSTRAR EJECUCIÓN EXTERNA (GRÁFICA) ---
        # Graficamos los primeros dos momentos de Hu para ver la separación
        plt.figure(figsize=(10, 6))
        for label in np.unique(y):
            mask = (y == label)
            plt.scatter(x[mask, 0], x[mask, 1], label=label)

        plt.title("Ejecución Externa: Espacio de Características (Momentos de Hu)")
        plt.xlabel("Momento de Hu 1")
        plt.ylabel("Momento de Hu 2")
        plt.legend()
        plt.grid(True)
        plt.show() # Esta línea abre la ventana externa

    else:
        print("No se encontraron imágenes. Revisa que las carpetas de clases estén dentro de 'dataset'.")
        