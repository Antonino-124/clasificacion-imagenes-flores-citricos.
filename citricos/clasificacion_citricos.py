import os
from pathlib import Path
import cv2
import numpy as np
import mahotas
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, hog
from skimage import exposure

# CONFIGURACIÓN Y FUNCIONES + RUTA DE CITRICOS

RUTA_DATASET = str(Path(__file__).resolve().parents[1] / "dataset" / "dataset" / "citricos")
NUM_PALABRAS = 100 

def orden_natural(lista):
    """Ordena las carpetas numéricamente (1, 2, 10 en lugar de 1, 10, 2)."""
    return sorted(lista, key=lambda f: int(re.sub(r'\D', '', f)) if re.sub(r'\D', '', f) else 0)

def preprocesar_imagen(img):
    """Normalización espacial y reducción de ruido."""
    img_res = cv2.resize(img, (256, 256))
    gris = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    gris_blur = cv2.GaussianBlur(gris, (3, 3), 0)
    return img_res, gris_blur

# LA GRÁFICA COMPARATIVA DE LOS MODELOS
def graficar_comparativa_modelos(resultados, titulo_desc):
    """Genera la gráfica de barras con los colores y estilo de la imagen de referencia."""
    modelos = list(resultados.keys())
    valores = list(resultados.values())
    # Colores: Azul para SVM, Rojo para k-NN, Verde para Random Forest
    colores = ['#3498db', '#e74c3c', '#2ecc71'] 

    plt.figure(figsize=(10, 6))
    barras = plt.bar(modelos, valores, color=colores)
    
    # Añadir el valor de accuracy sobre cada barra
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, yval + 0.01, f"{yval:.2f}", 
                 ha='center', va='bottom', fontweight='bold')

    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title(f'Barra Comparativa de los tres modelos ({titulo_desc})')
    plt.show()

def visualizar_analisis_completo(ruta, titulo_dataset, fotos_por_ventana=3):
    """Muestra Original, ORB Keypoints, LBP (Textura) y HOG (Estructura)."""
    if not os.path.exists(ruta): return
    
    carpetas = orden_natural([d for d in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, d))])
    muestras, nombres = [], []
    for n in carpetas:
        r = os.path.join(ruta, n)
        imgs = [f for f in os.listdir(r) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if imgs:
            muestras.append(os.path.join(r, imgs[0]))
            nombres.append(n)

    for i in range(0, len(muestras), fotos_por_ventana):
        lote_r = muestras[i : i + fotos_por_ventana]
        lote_n = nombres[i : i + fotos_por_ventana]
        num_fotos = len(lote_r)
        
        fig, axes = plt.subplots(num_fotos, 4, figsize=(18, 4 * num_fotos))
        fig.suptitle(f'{titulo_dataset} - Análisis de Descriptores (Grupo {i//fotos_por_ventana + 1})', fontsize=16, fontweight='bold')

        for j in range(num_fotos):
            img_raw = cv2.imread(lote_r[j])
            if img_raw is None: continue
            img_res, gris = preprocesar_imagen(img_raw)
            
            orb_detector = cv2.ORB_create(nfeatures=500)
            kp, _ = orb_detector.detectAndCompute(gris, None)
            img_orb = cv2.drawKeypoints(img_res, kp, None, color=(0, 255, 0))
            lbp = local_binary_pattern(gris, 8, 1, method="uniform")
            _, hog_img = hog(gris, orientations=9, pixels_per_cell=(16, 16), 
                             cells_per_block=(2, 2), visualize=True)
            hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

            axs = axes[j] if num_fotos > 1 else axes
            axs[0].imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)); axs[0].set_title(f"{lote_n[j]} (Original)"); axs[0].axis('off')
            axs[1].imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB)); axs[1].set_title("ORB Keypoints"); axs[1].axis('off')
            axs[2].imshow(lbp, cmap='gray'); axs[2].set_title("LBP (Textura)"); axs[2].axis('off')
            axs[3].imshow(hog_img_rescaled, cmap='viridis'); axs[3].set_title("HOG (Gradientes)"); axs[3].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# FUNCIONES DE EXTRACCIÓN

def extraer_globales(img_color, img_gris):
    hu = cv2.HuMoments(cv2.moments(img_gris)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-15)
    har = mahotas.features.haralick(img_gris).mean(axis=0)
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return np.hstack([hu, har, hist.flatten()])

def extraer_locales_crudos(img_color):
    img_res = cv2.resize(img_color, (128, 128))
    gris = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    _, orb_desc = cv2.ORB_create(nfeatures=500).detectAndCompute(gris, None)
    lbp = local_binary_pattern(gris, 8, 1, method="uniform")
    lbp_h, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10), density=True)
    h_desc = hog(gris, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return orb_desc, lbp_h, h_desc

def imprimir_tabla_final(res_g, res_l, res_t):
    linea = "=" * 80
    print(f"\n{linea}")
    print(f"{'MODELO':<15} | {'GLOBAL (%)':<15} | {'LOCAL (%)':<15} | {'COMBINADO (%)':<15}")
    print("-" * 80)
    for mod in res_g.keys():
        g = f"{res_g[mod]*100:>12.2f}%"
        l = f"{res_l[mod]*100:>12.2f}%"
        c = f"{res_t[mod]*100:>12.2f}%"
        print(f"{mod:<15} | {g} | {l} | {c}")
    print(f"{linea}\n")

def evaluar_detallado(X_data, y_data, titulo):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    sc = StandardScaler(); X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)
    modelos = {"SVM": SVC(kernel='linear'), "k-NN": KNeighborsClassifier(n_neighbors=5), 
               "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)}
    resultados_acc = {}
    print(f"\n{'='*65}\n REPORTE: {titulo}\n{'='*65}")
    for n, mod in modelos.items():
        mod.fit(X_train, y_train); y_pred = mod.predict(X_test)
        print(f"\n>>> MODELO: {n}\n{classification_report(y_test, y_pred, target_names=np.unique(y_data))}")
        resultados_acc[n] = accuracy_score(y_test, y_pred)
    return resultados_acc

# EJECUCIÓN

if not os.path.isdir(RUTA_DATASET):
    raise FileNotFoundError(
        f"No se encontró el dataset de cítricos en: {RUTA_DATASET}. "
        "Consulta dataset/README.md para organizarlo."
    )

print(">>> Generando análisis visual de descriptores...")
visualizar_analisis_completo(RUTA_DATASET, "Dataset Cítricos")

X_glob, X_orb_raw, X_loc_feat, y_list = [], [], [], []
carpetas = orden_natural([d for d in os.listdir(RUTA_DATASET) if os.path.isdir(os.path.join(RUTA_DATASET, d))])

print(f"\n>>> Extrayendo características en: {RUTA_DATASET}")
for nombre in carpetas:
    ruta_f = os.path.join(RUTA_DATASET, nombre)
    print(f"Procesando categoría: {nombre}")
    for archivo in os.listdir(ruta_f):
        img_raw = cv2.imread(os.path.join(ruta_f, archivo))
        if img_raw is not None:
            img_p, gris_p = preprocesar_imagen(img_raw)
            orb, lbp, h_hog = extraer_locales_crudos(img_p)
            if orb is not None:
                X_glob.append(extraer_globales(img_p, gris_p))
                X_orb_raw.append(orb)
                X_loc_feat.append(np.hstack([lbp, h_hog]))
                y_list.append(nombre)

X_global = np.array(X_glob); y = np.array(y_list)

#EVALUACIÓN Y GRÁFICA DE GLOBALES 
res_g = evaluar_detallado(X_global, y, "GLOBALES (Hu, Haralick, Hist)")
graficar_comparativa_modelos(res_g, "Globales")

#CONTINUACIÓN DEL PROCESO
print("\n>>> Entrenando K-Means para BoVW...")
kmeans = KMeans(n_clusters=NUM_PALABRAS, random_state=42, n_init=10).fit(np.vstack(X_orb_raw).astype(float))
X_bovw = []
for desc in X_orb_raw:
    hist, _ = np.histogram(kmeans.predict(desc.astype(float)), bins=np.arange(NUM_PALABRAS + 1))
    norm = np.linalg.norm(hist)
    X_bovw.append(hist/(norm + 1e-7) if norm > 0 else hist)

X_local = np.hstack([np.array(X_bovw), np.array(X_loc_feat)])
X_total = np.hstack([X_global, X_local])

res_l = evaluar_detallado(X_local, y, "LOCALES (ORB, LBP, HOG)")
res_t = evaluar_detallado(X_total, y, "FUSIÓN GLOBALES + LOCALES")

imprimir_tabla_final(res_g, res_l, res_t)

# Gráficas Finales
plt.figure(figsize=(10, 6))
for cat in np.unique(y):
    mask = (y == cat); plt.scatter(X_global[mask, 0], X_global[mask, 1], label=cat, alpha=0.6)
plt.title("Espacio de Características (Hu Moments)"); plt.xlabel("H1"); plt.ylabel("H2"); plt.legend(); plt.show()

etiquetas = list(res_g.keys()); x = np.arange(len(etiquetas)); width = 0.25
plt.figure(figsize=(10, 6))
plt.bar(x - width, res_g.values(), width, label='Globales', color='skyblue')
plt.bar(x, res_l.values(), width, label='Locales', color='salmon')
plt.bar(x + width, res_t.values(), width, label='Fusión', color='lightgreen')
plt.xticks(x, etiquetas); plt.ylabel('Accuracy'); plt.title('Desempeño Comparativo'); plt.legend(); plt.show()