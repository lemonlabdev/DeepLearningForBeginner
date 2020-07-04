import tensorflow as tf

# 세션이나 플레이스홀더를 사용하지 않음
# 어떤 전역 컬렉션도 참조하지 않고 규제를 직접 계산

# tf.zeros() - 모든 원소의 값이 0인 텐서를 생성
# tf.ones() - 모든 원소의 값이 1인 텐서를 생성

W = tf.Variable(tf.ones(shape=(2,2)), name="W") # 하나의 변수는 하나의 파이썬 객체
b = tf.Variable(tf.zeros(shape=(2)), name="b")

# 아래의 데코레이터를 사용하면 그래프 내에서 컴파일 되었을 때 더 빠르게 실행
# GPU나 TPU를 사용해 작동, SaveModel로 내보내는게 가능함

@tf.function
def forward(x):
    return W * x + b # 변수가 수식에 사용될 때, 자동적으로 tf.Tensor로 변환됨

out_a = forward([1,0])
print(out_a)