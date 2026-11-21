import cv2
import numpy as np
import matplotlib.pyplot as plt

#    Cargar imágenes
img1 = cv2.imread("rosa.jpeg")
img2 = cv2.imread("girasol.jpeg")
img3 = cv2.imread("tulipan.jpeg")

# Verificación de seguridad: Si una imagen no carga, el código fallará
if img1 is None or img2 is None or img3 is None:
    print("Error: No se encontró alguna de las imágenes. Revisa los nombres de archivo.")
else:
    # Convertir a espacio HSV
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hsv3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)

    # Calcular y Normalizar histogramas
    hist1 = cv2.calcHist([hsv1], [0], None, [180], [0,180])
    hist2 = cv2.calcHist([hsv2], [0], None, [180], [0,180])
    hist3 = cv2.calcHist([hsv3], [0], None, [180], [0,180])
    
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    cv2.normalize(hist3, hist3)
    
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()
    hist3 = hist3.flatten()

    # Muestra las imagenes
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)) # Convertir a RGB para que se vea bien en Matplotlib
    plt.title("Rosa")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Girasol")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.title("Tulipán")
    plt.axis('off')
    
    plt.tight_layout()
    

    # Grafica los histogramas
    plt.figure() 
    plt.plot(hist1, label="Rosa", color='red')
    plt.plot(hist2, label="Girasol", color='orange')
    plt.plot(hist3, label="Tulipán", color='magenta')
    plt.legend()
    plt.title("Histogramas normalizados (Canal H)")
    
    plt.show() # Esta es la instrucción que abre las ventanas externas