import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises
img = cv2.imread("Flor_Escala_de_Grises.jpg", cv2.IMREAD_GRAYSCALE)

# Ver forma
print("Shape:", img.shape)

# 1. Calcular promedio y desviación estándar
mean_pixel = img.mean()
std_pixel = img.std()
print("Promedio:", mean_pixel)
print("Desviación estándar:", std_pixel)

# 2. Calcular histograma de intensidades
hist = cv2.calcHist([img], [0], None, [256], [0,256])

# Muestra la imagen original
plt.figure() # Crea una ventana nueva para la imagen
plt.imshow(img, cmap='gray') 
# -------------------------------------------

# Mostrar histograma
plt.figure() # Crea otra ventana para el histograma
plt.plot(hist)
plt.title("Histograma de intensidades")
plt.xlabel("Nivel de gris")
plt.ylabel("Frecuencia")
plt.show()