import cv2
import numpy as np


def extract_orb_features(image_gray, n_features=500):
    # ORB es rápido y eficiente (libre de patentes)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    return keypoints, descriptors

def extract_sift_features(image_gray):
    # SIFT es más preciso pero más pesado computacionalmente
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, method='ORB'):
    if method == "ORB":
        # Para descriptores binarios como ORB se usa NORM_HAMMING
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif method == "SIFT":
        # Para SIFT se usa NORM_L2 (distancia euclidiana)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        raise ValueError("Método no soportado. Usa 'ORB' o 'SIFT'.")
    
    matches = matcher.match(desc1, desc2)
    # Ordenar por distancia (los mejores matches primero)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def compare_images_local(img_path1, img_path2, method='ORB'):
    # Leer imágenes
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None or img2 is None:
        print("Error: No se pudo cargar una de las imágenes.")
        return None

    # Convertir a escala de grises
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Extraer según el método seleccionado
    if method == "ORB":
        kp1, desc1 = extract_orb_features(gray1)
        kp2, desc2 = extract_orb_features(gray2)
    elif method == "SIFT":
        kp1, desc1 = extract_sift_features(gray1)
        kp2, desc2 = extract_sift_features(gray2)

    if desc1 is None or desc2 is None:
        print("No se detectaron descriptores en alguna imagen.")
        return None

    # Realizar el matching
    matches = match_descriptors(desc1, desc2, method)
    
    # Calcular distancia promedio como medida de similitud
    avg_distance = np.mean([m.distance for m in matches])
    
    return matches, avg_distance, kp1, kp2, img1, img2

img1_path = 'Rosa.jpeg' 
img2_path = 'Tulipan.jpeg'

# Ejecutar la comparación
result = compare_images_local(img1_path, img2_path, method="ORB")

if result:
    matches, avg_dist, kp1, kp2, img1, img2 = result
    
    print(f"Distancia promedio: {avg_dist:.2f}")
    print(f"Número de matches: {len(matches)}")

    # Dibujar las primeras 30 coincidencias
    img_matches = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        matches[:30], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Mostrar resultado
    cv2.imshow("Coincidencias (Matching)", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    