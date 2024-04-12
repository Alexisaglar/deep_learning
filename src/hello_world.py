import tensorflow as tf
import numpy as np
import pandas as pd


def get_minibatch(batch_size=32, data_size=784):
    # This is a placeholder implementation.
    # You should replace this with code to fetch your actual data.
    return tf.convert_to_tensor(np.random.rand(batch_size, data_size), dtype=tf.float32)


# Define the model as a function
@tf.function
def model(x):
    return tf.matmul(x, W) + b


# Initialize x, W and B
W = tf.Variable(tf.random.uniform([784, 10], -1, 1), name="W")
b = tf.Variable(tf.zeros([10]), name="biases")

x_batch = get_minibatch()

output = model(x_batch)

print(output)
