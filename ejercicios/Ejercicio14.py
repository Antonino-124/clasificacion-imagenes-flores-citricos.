import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- PARÁMETROS ---
DATASET_PATH = "dataset"
NUM_CLUSTERS = 100  # Tamaño del vocabulario visual
DESCRIPTOR = "ORB"  # Puede ser "ORB" o "SIFT"

# --- EXTRACCIÓN DE DESCRIPTORES ---
def extract_local_descriptors(image_path, method="ORB"):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "ORB":
        extractor = cv2.ORB_create(nfeatures=500)
    elif method == "SIFT":
        extractor = cv2.SIFT_create()
    else:
        raise ValueError("Método no válido")

    # Extraemos solo los descriptores (los puntos clave no se usan para el entrenamiento)
    _, descriptors = extractor.detectAndCompute(gray, None)
    return descriptors

# --- CARGA DEL DATASET ---
def load_dataset(path):
    descriptors_list = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists(path):
        print(f"Error: No se encuentra la carpeta {path}")
        return None, None, None

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        
        if not os.path.isdir(folder_path):
            continue

        label_map[current_label] = folder
        print(f"Procesando clase: {folder}...")

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            desc = extract_local_descriptors(img_path, DESCRIPTOR)
            
            if desc is not None:
                descriptors_list.append(desc)
                labels.append(current_label)
        
        current_label += 1
    
    return descriptors_list, labels, label_map

# --- CONSTRUCCIÓN DE BAG OF VISUAL WORDS (BoVW) ---
def build_bovw(descriptor_list, kmeans):
    features = []
    for desc in descriptor_list:
        # Inicializamos un histograma de tamaño igual al número de clusters
        histogram = np.zeros(NUM_CLUSTERS)
        
        # Predecimos a qué cluster pertenece cada descriptor de la imagen
        clusters = kmeans.predict(desc)
        
        # Llenamos el histograma (conteo de palabras visuales)
        for c in clusters:
            histogram[c] += 1
        
        # Normalizamos el histograma (L2 norm) para que no dependa de la cantidad de puntos
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
            
        features.append(histogram)
    
    return np.array(features)

# --- FLUJO PRINCIPAL (MAIN) ---
if __name__ == "__main__":
    # Cargar descriptores de todas las imágenes
    descriptors, labels, label_map = load_dataset(DATASET_PATH)

    if descriptors:
        # Entrenar KMeans para crear el "Vocabulario Visual"
        print("\nEntrenando KMeans (Creando vocabulario)...")
        # Unimos todos los descriptores de todas las imágenes en una sola matriz gigante
        all_descriptors = np.vstack(descriptors)
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
        kmeans.fit(all_descriptors)

        # Construir vectores de características (Histogramas BoVW)
        print("Construyendo vectores BoVW...")
        X = build_bovw(descriptors, kmeans)
        y = np.array(labels)

        # Dividir datos en Entrenamiento y Prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el Clasificador (SVM)
        print("Entrenando clasificador SVM...")
        svm = SVC(kernel="linear", probability=True)
        svm.fit(X_train, y_train)

        # Evaluación final
        print("\n--- RESULTADOS ---")
        y_pred = svm.predict(X_test) 
        print(classification_report(y_test, y_pred, target_names=list(label_map.values())))
    else:
        print("No se pudieron cargar los datos.")