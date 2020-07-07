import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
- x_data가 x1과 x2를 기준으로 y_data 0과 1로 구분하는 예제입니다
- Logistic Classification 통해 보라색과 노란색 y_data(Label)을 구분해 보겠습니다.
- Test 데이터는 붉은색의 위치와 같이 추론시 1의 값을 가지게 됩니다.
"""

x_train = [[1., 2.],
           [2., 3.],
           [3., 1.],
           [4., 3.],
           [5., 3.],
           [6., 2.]]
y_train = [[0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.]]

x_test = [[5., 2.]]
y_test = [[1.]]

x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]

colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1, x2, c=colors, marker='^')
plt.scatter(x_test[0][0], x_test[0][1], c="red", marker="x")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Batch Size 는 한번에 학습시킬 Size
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))  # .repeat()

W = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')


def logistic_regression(features):
    # tf.exp = exponential function
    # Sigmoid 함수를 마지막 레이어의 활성함수로 사용
    hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis


def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features)) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy


def grad(features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features),features,labels)
    return tape.gradient(loss_value, [W, b])

EPOCHS = 1001

for step in range(EPOCHS):
    for features, labels  in iter(dataset):
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels)))
test_acc = accuracy_fn(logistic_regression(x_test),y_test)
print("Testset Accuracy: {:.4f}".format(test_acc))
