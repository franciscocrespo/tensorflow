import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

filas = 100
columnas = 3

# Inicializa una matriz de 5 por 10 con 0s. Esta forma de hacerlo en python en listas por comprension
matrix = [[0]*columnas for i in range(filas)]

#trabajamos con tf
vectors = tf.constant(matrix)
# toma el vector y, como segundo argumento, toma el espacio que ocupara. En este caso (Dimension=1, Dimension=100, Dimension=3)
expanded_vectors = tf.expand_dims(vectors, 0)
print(expanded_vectors.get_shape())
