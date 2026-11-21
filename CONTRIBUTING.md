# Notas para reproducir o mejorar el proyecto

1. Instala dependencias en un entorno virtual.
2. Usa datasets cuyo uso y redistribución estén autorizados.
3. Ejecuta `python verificar_dataset.py` antes de entrenar.
4. No incluyas rutas absolutas, datos personales, archivos `__pycache__` ni modelos pesados.
5. Si modificas la evaluación, separa entrenamiento/prueba **antes** de ajustar K-Means, escaladores u otros estimadores.
6. Documenta los cambios de métricas y el origen de los datos.
