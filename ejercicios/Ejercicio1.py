import numpy as np

# Creación de arreglos básicos
vector = np.array([1, 2, 3])
matriz = np.array([[1, 2, 3], [4, 5, 6]])

# Creación de arreglos con funciones
print("Ceros:\n", np.zeros((3, 3)))
print("Unos:\n", np.ones((2, 2)))
print("Rango (0 a 10 de 2 en 2):", np.arange(0, 10, 2))
print("Aleatorios:\n", np.random.rand(3, 3))

# Operaciones aritméticas
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("\nSuma a + b:", a + b)
print("Multiplicación a * b:", a * b)
print("Potencia a al cuadrado:", a ** 2)

# Uso de índices
mat = np.array([[10, 20, 30], [40, 50, 60]])
print("\nElemento en [0,1]:", mat[0, 1])
print("Segunda columna completa:", mat[:, 1])

# Programación orientada a arreglos (Vectorización)
x = np.arange(1, 6)
y = x * 2
print("\nVector x:", x)
print("Vector y (x * 2):", y)

# Funciones universales
print("\nRaíz cuadrada de x:", np.sqrt(x))
print("Promedio de x:", np.mean(x))
print("Máximo de x:", np.max(x))
print("Seno de x:", np.sin(x))