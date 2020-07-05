import tensorflow as tf
import numpy as np

# Data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# W, b initialize
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# Set learning_rate
learning_rate = tf.constant(0.01, dtype=tf.float32)

# W, b update
for i in range(100):
    # TF2.x's Automatic Differentiation, it records all computation in context to tape
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    # tape.gradient(target, sources)
    W_grad, b_grad = tape.gradient(cost, [W, b])  # compute the gradient, using reverse mode differentiation
    W.assign_sub(learning_rate * W_grad)  # use to change a tf.Variable, this syntax means -=
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        # Tensor are explicitly converted to NumPy ndarrays using .numpy() method
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))


# predict
print(W * 5 + b)
print(W * 2.5 + b)
