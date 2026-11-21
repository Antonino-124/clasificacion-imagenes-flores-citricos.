# Datos para la clasificación de imágenes

Los scripts esperan las imágenes en las siguientes rutas relativas al repositorio:

```text
dataset/dataset/flores/Clase1 ... Clase17
dataset/dataset/citricos/clorosis
dataset/dataset/citricos/huanglongbing
dataset/dataset/citricos/mancha_foliar
dataset/dataset/citricos/mancha_grasienta
dataset/dataset/citricos/sanas
```

En el material de trabajo proporcionado se identificaron **1,364 imágenes de flores en 17 carpetas** y **1,000 imágenes de cítricos en 5 carpetas**. La carpeta `Clase14` contiene 84 imágenes y las demás carpetas de flores contienen 80.

Las **2,364 imágenes están incluidas** en `dataset/dataset/` para su ejecución directa. El publicador indica contar con permiso para compartirlas; no se asigna ninguna licencia nueva ni se presume titularidad sobre el material de terceros. Se recomienda documentar fuente, atribución y términos aplicables de cada conjunto.

Para comprobar la estructura, ejecuta:

```powershell
python verificar_dataset.py
```
