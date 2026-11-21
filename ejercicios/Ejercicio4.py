#Ejercicio 4. Carga y visualización de imágenes (Numpy + Matplotlib)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('Flor_Escala_de_Grises.jpg')  # imagen en escala de grises o RGB
plt.imshow(img)
plt.axis('off')
plt.show() #mostrar imagen en ventana emergente 