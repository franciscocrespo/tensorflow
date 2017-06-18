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
    # Si el número es mayor a 0.5
    if np.random.random() > 0.5:
        # Si es mayor a 0.5 creamos puntos con una random normal de media 0.0 y desviación 0.9
        conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

# Vector de tonsorflow con el conjunto de puntos
vectors = tf.constant(conjunto_puntos)
k = 4 # número de centroides
# Está tomando las 4 primeras posiciones del tensor.
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))
# Esto es para hacer igual a los tensores, si no son de dimensiones iguales no se podrán hacer operaciones entre ellosself.
expanded_vectors = tf.expand_dims(vectors, 0) # Añadimos una nueva dimension en x (0)
expanded_centroides = tf.expand_dims(centroides, 1) # Añadimos una nueva dimension en y (1)
# Subtract sabe las dimensiones que matchean, tras acomodar los tensores y calcula la diferencia |xi - centroidei|
diff = tf.subtract(expanded_vectors, expanded_centroides)
sqr = tf.square(diff) # Obtiene el cuadrado de |xi - centroidei|
distances = tf.reduce_sum(sqr, 2) # Suma el cuadrado de la difertncia (sqr) para cada punto (x, y Que es la dimensión 2)

assignments = tf.argmin(distances, 0) # Retorna el índice del elemento con el valor menor en la dimensión del tensor indicada
'''
Cálculo de los nuevos centroides
Una vez hemos creado nuevos grupos en cada iteración, recordemos que el paso siguiente del algoritmo consiste en calcular
los nuevos centroides de los nuevos grupos formados. En el código del apartado anterior se expresaba con esta línea de código:

means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in range(k)])

En este código, se observa que el tensor means es el resultado de concatenar k tensores que corresponden al valor medio de
todos los puntos pertenecientes a cada uno de los k clusters.

A continuación, comento cada una de las operaciones TensorFlow que intervienen en el cálculo del valor medio de los puntos
pertenecientes a un cluster. Creo que el siguiente nivel de explicación es suficiente para el propósito de este libro:

 - Con equal se obtiene un tensor booleano (Dimension(2000)) que indica (con valor “true” )
las posiciones donde el valor del tensor assignment coincide con el cluster c (uno de los k),
del que en aquel momento estamos calculando el valor medio de sus puntos.

 - Con where se construye un tensor (Dimension (1) x Dimension(2000)) con la posición donde
 se encuentran los valores “true” en el tensor booleano recibido como parámetro. Es decir,
 una lista con las posiciones de estos.

 - Con reshape se construye un tensor (Dimension (2000) x Dimension(1) ) con los índices
 de los puntos en el tensor vectors que pertenecen a este cluster c.

 - Con gather se construye un tensor (Dimension (1) x Dimension (2000) x Dimension(2) ) que
 reúne las coordenadas de los puntos que forman el cluster c.

 - Con tf.reduce_mean se construye un tensor (Dimension (1) x Dimension(2) ) que contiene el
 valor medio de todos los puntos que pertenecen a este cluster c.

 De todas formas, si el lector quiere profundizar un poco más en el código, como siempre indico, se puede encontrar más información
 para cada una de estas operaciones, con ejemplos muy aclaratorios, en la página de la API de TensorFlow[24]:
'''

means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in range(k)], 0)

# Actualizamos los centroides
update_centroides = tf.assign(centroides, means)
# inicializamos las variables
init_op = tf.initialize_all_variables()
# Creamos la sesion y arrancamos
sess = tf.Session()
sess.run(init_op)

for step in range(100):
   _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

print(centroid_values)

# Gráfico
data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignment_values)):
  data["x"].append(conjunto_puntos[i][0])
  data["y"].append(conjunto_puntos[i][1])
  data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()
