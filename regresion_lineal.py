import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_puntos = 1000
conjunto_puntos = []
for i in range(num_puntos):
    # X1 es un conjunto de datos aleatorio que sigue una distribución normal
    x1 = np.random.normal(0.0, 0.55)
    # y1 es la variable dependiente
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    conjunto_puntos.append([x1, y1])

# obtenemos los datos x e y del arrglo de conjunto de puntos
x_data = [v[0] for v in conjunto_puntos]
y_data = [v[1] for v in conjunto_puntos]

# Con esto dibujamos la salida, los datos originales
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()

# Comenzamos con el modelo para la regresión lineal
# Definimos dos variables que ocupan nodos de tf y las inicializamos
# [1] se refiere a un array de una dimensión
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # Argumentos random_uniform(shape, max value, min value)
b = tf.Variable(tf.zeros([1])) # Argumentos zeros(shape), inicializa en 0s
y = W * x_data + b

# función de coste (calculamos la media de la dicerencia al cuadrado)
loss = tf.reduce_mean(tf.square(y - y_data)) # Es el error, o la diferencia entre lo esperado, y_data, y lo obtenido, y.

# Algoritmo del gradiente decreciente. Hace que descienda el error de la función de coste.
optimizer = tf.train.GradientDescentOptimizer(0.5) # Instanciamos el algoritmo y proporcionamos un rate learning
train = optimizer.minimize(loss) # bajamos el error

# ----------------------- Comenzamos a ejecutar nuestro modelo ---------------------------------------------------

# Inicializamos las variables cargadas en el grafo de tensorflow
init = tf.global_variables_initializer()
# Creamos una session
sess = tf.Session()
# Arrancamos con las variables inicializadas
sess.run(init)

# Una vez que el modelo arranco empezamos con el prceso iterativo, en este caso usamos 8 iteraciones
for step in range(12):
    # Arrancamos con el algoritmo de disminución de costo
    sess.run(train)
    print(step, sess.run(W), sess.run(b))
    #Graphic display
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.xlabel('x')
    plt.xlim(-2,2)
    plt.ylim(0.1,0.6)
    plt.ylabel('y')
    plt.legend()
    plt.show()
