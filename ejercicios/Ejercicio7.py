from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


ruta = str(Path(__file__).resolve().parents[1] / "dataset" / "dataset")
X = []
y = []

# 1. Cargar imágenes y extraer características
for clase in os.listdir(ruta):
    carpeta = os.path.join(ruta, clase)
    if os.path.isdir(carpeta):
        for archivo in os.listdir(carpeta):
            img = cv2.imread(os.path.join(carpeta, archivo), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Características simples
                mean_pixel = img.mean()
                std_pixel = img.std()
                X.append([mean_pixel, std_pixel])
                y.append(clase)

X = np.array(X)
y = np.array(y)
print("Datos:", X.shape)

# 2. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Entrenar clasificador
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# 4. Evaluar
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- MOSTRAR EJECUCIÓN EXTERNA ---
for c in np.unique(y):
    plt.scatter(X[y == c, 0], X[y == c, 1], label=c)

plt.title("Clasificación SVM")
plt.xlabel("Media")
plt.ylabel("Desviación")
plt.legend()
plt.show()