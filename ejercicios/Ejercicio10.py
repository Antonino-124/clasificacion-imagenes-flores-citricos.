import cv2
import numpy as np
import mahotas
from scipy.spatial import distance

# Definición de los descriptores globales
def hu_moments(gray_img):
    # Calcula los 7 momentos de Hu (forma)
    moments = cv2.moments(gray_img)
    hu = cv2.HuMoments(moments).flatten()
    return hu

def haralick_features(gray_img):
    # Calcula texturas de Haralick
    har = mahotas.features.haralick(gray_img).mean(axis=0)
    return har

def color_histogram(img):
    # Histograma de color en espacio HSV (más robusto que BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Función principal de extracción (como en tu imagen)
def extract_features(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extracción de cada tipo de característica
    hu = hu_moments(gray)
    har = haralick_features(gray)
    color = color_histogram(img)
    
    # Unimos todo en un solo vector (Feature Vector)
    return np.hstack([hu, har, color])

#Comparación de imágenes
try:
    img1_feat = extract_features("Girasol.jpeg")
    img2_feat = extract_features("Rosa.jpeg")
    img3_feat = extract_features("Tulipan.jpeg")

    # Cálculo de Distancia Euclídea
    # Entre menor sea la distancia, más parecidas son las imágenes
    d12 = distance.euclidean(img1_feat, img2_feat)
    d13 = distance.euclidean(img1_feat, img3_feat)

    print(f"Distancia girasol vs rosa: {d12:.4f}")
    print(f"Distancia girasol vs tulipan:  {d13:.4f}")

    if d12 < d13:
        print("\nResultado: el girasol es más parecida a rosa que al tulipan.")
    else:
        print("\nResultado: Curiosamente, el girasol se parece más al tulipan.")

except Exception as e:
    print(f"Error en la ejecución: {e}")
    