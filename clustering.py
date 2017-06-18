import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Crearemos los datos para el modelo. 2000 puntos random con 2 distribuciones normales.
numero_puntos = 2000;
conjunto_puntos = []

for i in range(numero_puntos):
    # &gt es equivalente a >
    if np.random.random() > 0.5:
        # Si es mayor a 0.5 creamos puntos con una random normal de media 0.0 y desviación 0.9
        conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

# Grafico
df = pd.DataFrame({"x": [v[0] for v in conjunto_puntos],
        "y": [v[1] for v in conjunto_puntos]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()
