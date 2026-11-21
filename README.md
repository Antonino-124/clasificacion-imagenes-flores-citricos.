# Clasificación de imágenes de flores y hojas de cítricos

## Descripción

Proyecto académico de visión por computadora tradicional y aprendizaje automático (*Machine Learning*) que compara la clasificación de dos conjuntos de imágenes: 17 categorías de flores y cinco categorías de hojas de cítricos (hojas sanas y cuatro condiciones: clorosis, huanglongbing, mancha foliar y mancha grasienta).

El flujo de trabajo incluye preprocesamiento, extracción de características globales y locales, construcción de un vocabulario visual con *Bag of Visual Words* (BoVW) y comparación de tres clasificadores: Support Vector Machine (SVM), k-Nearest Neighbors (k-NN) y Random Forest.

> Este proyecto utiliza técnicas tradicionales de visión por computadora; no implementa redes neuronales profundas.

## Objetivos

- Preparar imágenes para su análisis.
- Extraer características de forma, textura y color.
- Obtener descriptores locales y representarlos mediante BoVW.
- Comparar clasificadores usando características globales, locales y combinadas.
- Evaluar el comportamiento de los modelos en dos problemas de clasificación diferentes.

## Flujo de trabajo

1. **Preprocesamiento:** redimensionamiento a 256 × 256 píxeles, conversión a escala de grises y filtrado Gaussiano.
2. **Características globales:** Momentos de Hu, descriptores de Haralick y histogramas de color en espacio HSV.
3. **Características locales:** ORB, Local Binary Patterns (LBP) y Histogram of Oriented Gradients (HOG).
4. **Bag of Visual Words:** agrupamiento de descriptores ORB mediante K-Means para construir histogramas de palabras visuales.
5. **Clasificación:** entrenamiento y comparación de SVM, k-NN y Random Forest.
6. **Evaluación:** Accuracy y reportes de clasificación con Precision, Recall y F1-score.

## Estructura del repositorio

```text
clasificacion-imagenes-flores-citricos/
├── flores/
│   └── clasificacion_flores.py
├── citricos/
│   └── clasificacion_citricos.py
├── ejercicios/
│   ├── Ejercicio1.py
│   ├── ...
│   └── Ejercicio14.py
├── dataset/
│   ├── README.md
│   └── dataset/
│       ├── flores/     # 17 clases, 1,364 imágenes
│       └── citricos/   # 5 clases, 1,000 imágenes
├── documentacion/
│   └── Practica_Unidad_1_Resultados.pdf
├── resultados/
├── requirements.txt
└── .gitignore
```

Los scripts de `ejercicios/` corresponden a prácticas complementarias de procesamiento de imágenes y descriptores. Los dos programas principales están en `flores/` y `citricos/`. `verificar_dataset.py` revisa la existencia de las carpetas de imágenes sin requerir instalar bibliotecas de Machine Learning.

## Requisitos

- Python 3.9 o posterior (recomendado).
- Las bibliotecas indicadas en `requirements.txt`.
- Los conjuntos de imágenes organizados como se describe en `dataset/README.md`.

## Instalación

Clona el repositorio e instala las dependencias:

```bash
git clone URL_DEL_REPOSITORIO
cd clasificacion-imagenes-flores-citricos
python -m pip install -r requirements.txt
```

## Preparación de los datos

Los datasets ya están incluidos. Los programas buscan las imágenes en:

- `dataset/dataset/flores/`
- `dataset/dataset/citricos/`

Dentro de cada carpeta deben existir subcarpetas, una por categoría. Consulta `dataset/README.md` para ver la estructura esperada.

Esta versión incluye los **2,364 archivos de imágenes** de los conjuntos de flores y cítricos. El autor que publica el repositorio debe contar con los permisos necesarios de distribución. Ver `dataset/README.md`.

## Ejecución

Desde la carpeta raíz del proyecto:

```bash
python flores/clasificacion_flores.py
```

Para el conjunto de cítricos:

```bash
python citricos/clasificacion_citricos.py
```

Antes de ejecutarlos, verifica la estructura con `python verificar_dataset.py`. Los programas muestran visualizaciones, reportes de clasificación y gráficas comparativas durante la ejecución. Para evitar bloquear la ejecución, ciérralas según aparezcan.

## Resultados documentados

El PDF incluido en `documentacion/` contiene el desarrollo de la práctica, las visualizaciones y el análisis de resultados entregado como evidencia académica.

En el análisis escrito del informe se reporta que:

- Para flores, la combinación de descriptores globales y locales ofrece el mejor comportamiento descrito, con resultados alrededor del 69%.
- Para hojas de cítricos, el informe describe resultados cercanos al 97% con Random Forest utilizando descriptores globales.
- Random Forest se identifica en el informe como el clasificador con mejor desempeño general en las pruebas realizadas.

Estos valores se presentan como resultados consignados en el informe académico original; no se han vuelto a calcular en esta versión del repositorio.

## Herramientas y bibliotecas

Python · OpenCV · NumPy · Mahotas · Matplotlib · scikit-learn · scikit-image · SciPy · pandas

## Contexto académico

Desarrollado como práctica de la asignatura relacionada con procesamiento y análisis de datos, con el propósito de aplicar técnicas de visión por computadora y aprendizaje automático a problemas de clasificación de imágenes.

## Notas de reproducibilidad y mejoras pendientes

- Los resultados documentados son **históricos**: no se han vuelto a calcular con esta preparación.
- La construcción del vocabulario visual con K-Means sobre todos los descriptores se realiza antes de la separación de entrenamiento/prueba. Esto introduce una posible **fuga de información** en las evaluaciones que utilizan BoVW; las métricas originales deben interpretarse con esa limitación. Para una evaluación rigurosa, ajustar K-Means únicamente con entrenamiento y transformar después el conjunto de prueba.
- Los programas se ejecutan como scripts y abren múltiples gráficos interactivos; requieren un entorno Python con interfaz gráfica para visualizar todo el análisis.
- El informe académico original se conserva en el paquete local, pero no en el paquete público por contener información personal.

## Publicación responsable

No subas imágenes del dataset ni informes con datos personales a un repositorio público hasta confirmar las autorizaciones correspondientes. El `.gitignore` excluye las imágenes originales de los commits.

## Autoría

Proyecto académico de análisis y clasificación de imágenes. Personaliza esta sección con tu nombre público y el enlace a tu perfil de GitHub antes de publicarlo.


## Publicación en GitHub

El repositorio contiene 2,364 imágenes dentro de `dataset/dataset/`. Antes de hacer público el proyecto, confirma que tienes autorización para redistribuir ambos conjuntos y agrega las referencias y términos de licencia de las fuentes cuando estén disponibles. El PDF académico con datos personales no se incluye.

Para cargarlo en el repositorio existente, copia el contenido de esta carpeta (no el ZIP comprimido) dentro de tu clon local; después ejecuta `git add .`, `git commit -m "Publicar proyecto con datasets"` y `git push origin HEAD`.
