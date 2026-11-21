"""Comprueba la presencia y distribución del dataset sin instalar dependencias externas."""
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent / "dataset" / "dataset"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ESPERADOS = {"flores": 17, "citricos": 5}

def main():
    total = 0
    errores = []
    for conjunto, clases_esperadas in ESPERADOS.items():
        carpeta = RAIZ / conjunto
        if not carpeta.is_dir():
            errores.append(f"Falta el directorio: {carpeta}")
            continue
        clases = sorted(p for p in carpeta.iterdir() if p.is_dir())
        print(f"\n{conjunto.upper()}: {len(clases)} clases (esperadas: {clases_esperadas})")
        if len(clases) != clases_esperadas:
            errores.append(f"{conjunto}: se esperaban {clases_esperadas} clases")
        for clase in clases:
            cantidad = sum(f.is_file() and f.suffix.lower() in EXTENSIONES for f in clase.iterdir())
            print(f"  {clase.name}: {cantidad} imagenes")
            total += cantidad
            if cantidad == 0:
                errores.append(f"Clase vacia: {clase}")
    print(f"\nTotal de imagenes: {total}")
    if errores:
        print("\nADVERTENCIAS:")
        for error in errores: print(f"- {error}")
        raise SystemExit(1)
    print("Estructura de datos verificada correctamente.")

if __name__ == "__main__":
    main()
