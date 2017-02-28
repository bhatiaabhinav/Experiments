import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "pics/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3])
yo = tf.slice(image, [1000, 0, 0], [3000, -1, -1])

with tf.Session() as s:
    result = s.run(yo, feed_dict={image:raw_image_data})
    plt.imshow(result)
    plt.show()