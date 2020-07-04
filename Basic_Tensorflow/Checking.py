import tensorflow as tf

print(tf.__version__)

# 상수 만들기
constNumber = tf.constant(3.0, dtype=tf.float32) # value는 3.0. 데이터 타입은 tf.float32
print(constNumber)

# Tensor
constTensor0 = tf.constant(3) # a rank 0 tensor; this is a scalar with shape []
constTensor1 = tf.constant([1. ,2., 3.]) # a rank 1 tensor; this is a vector with shape [3]
constTensor2 = tf.constant([[1., 2., 3.], [4., 5., 6.]]) # a rank 2 tensor; a matrix with shape [2, 3]
constTensor3 = tf.constant([[[1., 2., 3.]], [[7., 8., 9.]]]) # a rank 3 tensor with shape [2, 1, 3]

print(constTensor0, constTensor1, constTensor2, constTensor3, sep='\n') # Shape의 변화에 집중하자