import numpy as np
import tensorflow as tf

X = tf.constant(np.eye(10000))
Y = tf.constant(np.random.randn(10000, 300))
Z = tf.matmul(X, Y)

with tf.Session() as s:
    model = tf.global_variables_initializer()
    s.run(model)
    s.run(Z)