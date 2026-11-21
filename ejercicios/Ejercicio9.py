import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Función para extraer Momentos de Hu (Forma)
def hu_moments(image_gray):
    moments = cv2.moments(image_gray)
    hu = cv2.HuMoments(moments).flatten()
    # Se aplica logaritmo para que los valores sean más fáciles de manejar
    return hu

# Función para extraer Características de Haralick (Textura)
def haralick_features(image_gray):
    # Creamos la matriz de co-ocurrencia (GLCM)
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], 
                        levels=256, symmetric=True, normed=True)
    
    # Extraemos propiedades de textura
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return np.array([contrast, energy, homogeneity, correlation])

# Función para extraer Histograma de Color
def color_histogram(image):
    # Calculamos histograma en los 3 canales (RGB) con 8 contenedores por canal
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# --- EJECUCIÓN PRINCIPAL ---

# Cargar la imagen ('flor.jpg' esté en la misma carpeta)
img_bgr = cv2.imread("Rosa.jpeg")

if img_bgr is None:
    print("Error: No se encontró la imagen 'flor.jpg'.")
else:
    # Convertir a escala de grises para Hu y Haralick
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Extraer todos los descriptores
    hu = hu_moments(gray)
    har = haralick_features(gray)
    color = color_histogram(img_bgr)

    # Unir todo en un solo vector (Concatenación)
    feature_vector = np.hstack([hu, har, color])

    # Mostrar resultados
    print("-" * 30)
    print("RESULTADOS DE EXTRACCIÓN")
    print("-" * 30)
    print(f"Momentos de Hu (Forma): {hu.shape[0]} valores")
    print(f"Haralick (Textura): {har.shape[0]} valores")
    print(f"Histograma (Color): {color.shape[0]} valores")
    print("-" * 30)
    print(f"Tamaño total del vector: {feature_vector.shape}")
    print("-" * 30)
    print("Vector de características final:")
    print(feature_vector)