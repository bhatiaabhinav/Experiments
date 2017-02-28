import tensorflow as tf
import numpy as np

c = tf.Variable(0, name='c')
s = tf.Variable(0, name='s')
y = tf.Variable(0, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("YO", session.graph)
    for i in range(10):
        c = c + 1
        print(session.run(c))
        print(session.run(c))
        s = s + np.random.randint(1000)
        print(session.run(s))
        print(session.run(s))
        y = s / c
        print(session.run(y))
        print(session.run(y))