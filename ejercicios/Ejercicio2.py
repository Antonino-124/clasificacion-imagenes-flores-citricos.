import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10)
y = x**2

# Mostrar resultados en consola
print("Valores de X:")
print(x)

print("\nValores de Y (X al cuadrado):")
print(y)

# Figuras y subplots
plt.figure()
plt.plot(x, y)
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].plot(x, y)
axs[1].plot(x, x)
plt.show()

# Gestión de gráficas
plt.title("Grafica")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# Guardado de gráficas
plt.savefig("grafica.png")
