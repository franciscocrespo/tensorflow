import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Variables simbólicas, asi se definen las variables en tensorflow.
a = tf.placeholder("float")
b = tf.placeholder("float")

# Función para multiplicar las variables a y b
y = tf.multiply(a, b)

# El programa se ejecuta dentro de la sesión, antes no.
sess = tf.Session()

# En este caso se pasa la operacion "y" con los parametros "a" y "b".
print(sess.run(y, feed_dict={a: 3, b: 3}))
