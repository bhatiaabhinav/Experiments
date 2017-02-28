from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W) + b

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

errors = []

with tf.Session() as sess:
    sess.run(init)

    for i in range(30000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _,e = sess.run([train_step, cross_entropy], feed_dict={x:batch_xs, y_:batch_ys})
        errors.append(e)
        print(e)

    #now test the model:
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    print("Test accuracy")
    acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}) * 100
    print(str(acc) + "%")
    
plt.plot(errors)
plt.show()