import tensorflow as tf
import pandas as pd
import numpy as np
import datetime


train_data = pd.read_csv("data/mnist_train.csv", header=None)
test_data = pd.read_csv("data/mnist_test.csv", header=None)

# test data
test_images = test_data.iloc[:, 1:].values
test_labels = test_data.iloc[:, 0].values

# train data
train_images = train_data.iloc[:, 1:].values
train_labels = train_data.iloc[:, 0].values

# normalize and flatten data
train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)


# initialize weights and bias
W = tf.Variable(tf.random.normal([784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([10]), name="biases")


# Convert labels to one-hot encoding
train_labels = tf.one_hot(train_labels, depth=10)
test_labels = tf.one_hot(test_labels, depth=10)


# define the model function
def model(x):
    return tf.matmul(x, W) + b


# loss function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )


# set up TensorBoard loggin
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# define the optimiser
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# training parameters
num_epochs = 10
batch_size = 100
num_batches = train_images.shape[0] // batch_size

# training loop
for epoch in range(num_batches):
    for batch_index in range(num_batches):
        batch_start = batch_size * batch_index
        batch_end = batch_size + batch_index
        x_batch = train_images[batch_start:batch_end]
        y_batch = train_labels[batch_start:batch_end]

        with tf.GradientTape() as tape:
            logits = model(x_batch)
            loss = compute_loss(logits, y_batch)

        gradient = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradient, [W, b]))

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=epoch * num_batches + batch_index)
            tf.summary.histogram(
                "logits", logits, step=epoch * num_batches + batch_index
            )
            tf.summary.histogram("weights", W, step=epoch * num_batches + batch_index)
            tf.summary.histogram("biases", b, step=epoch * num_batches + batch_index)
    print("epoch")
print(f"Epoch {epoch +1}, Loss:{loss.numpy()}")
