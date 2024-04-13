import tensorflow as tf
import numpy as np
import datetime

# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize and flatten the images
train_images = (train_images.reshape([-1, 784]) / 255.0).astype(np.float32)
test_images = (test_images.reshape([-1, 784]) / 255.0).astype(np.float32)

# Convert labels to one-hot encoding
train_labels = tf.one_hot(train_labels, depth=10)
test_labels = tf.one_hot(test_labels, depth=10)

# Initialize weights and biases
W = tf.Variable(tf.random.normal([784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([10]), name="biases")


# Define the model function
def model(x):
    return tf.matmul(x, W) + b


# Loss function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )


# Set up logging for TensorBoard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Define the optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# Training parameters
num_epochs = 10
batch_size = 100
num_batches = train_images.shape[0] // batch_size

# Training loop
for epoch in range(num_epochs):
    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = batch_start + batch_size
        x_batch = train_images[batch_start:batch_end]
        y_batch = train_labels[batch_start:batch_end]

        with tf.GradientTape() as tape:
            logits = model(x_batch)
            loss = compute_loss(logits, y_batch)

        gradients = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=epoch * num_batches + batch_index)
            tf.summary.histogram(
                "logits", logits, step=epoch * num_batches + batch_index
            )
            tf.summary.histogram("weights", W, step=epoch * num_batches + batch_index)
            tf.summary.histogram("biases", b, step=epoch * num_batches + batch_index)

    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# Run TensorBoard
# %load_ext tensorboard
# %tensorboard --logdir logs/gradient_tape
