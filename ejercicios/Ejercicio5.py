import cv2
import numpy as np

# Cargar imagen
img = cv2.imread("Imagen_Flor.jpg")  # cambia por tu ruta

# 1. Mostrar forma de la imagen (shape)
print("Imagen_Flor.jpg:", img.shape) 

# (alto, ancho, canales)
# 2. Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Recortar una región (ROI)
# Ejemplo: recorte desde (y1:y2, x1:x2)
roi = img[100:300, 150:400]

# 4. Cambiar brillo y contraste
# alpha = contraste (1.0 = sin cambio)
# beta = brillo (0 = sin cambio)
alpha = 1.5   # contraste
beta = 50     # brillo
brillo_contraste = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Mostrar resultados
cv2.imshow("Original", img)
cv2.imshow("Grises", gray)
cv2.imshow("Recorte", roi)
cv2.imshow("Brillo y Contraste", brillo_contraste)
cv2.waitKey(0)
cv2.destroyAllWindows()