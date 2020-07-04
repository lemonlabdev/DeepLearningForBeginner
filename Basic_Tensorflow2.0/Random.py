import tensorflow as tf

"""
텐서플로우는 tf.random 모듈에서 유사 난수 생성기를 제공, 두가지 방식을 제공함 

1. tf.random.Generator 
- 객체 사용을 통한 방식. 각 객체는 상태를(tf.Variable)안에 유지
- 이 상태는 매 숫자 생성때마다 변화
2. tf.random.stateless_uniform 
- 순수-함수형으로 상태가 없는 랜덤함수를 통한 방식
- 같은 디바이스에서 동일한 인수(시드값 등)을 통해 해당함수를 호출 시 항상 같은 결과를 출력

* 주의
tf.random.uniform, tf.random.normal 같은 구버전 TF 1.x의 RNG 들은 사용 권장 x
"""

# 분산 전략을 위해 두 개의 가상 디바이스 cpu:0, cpu:1 을 생성
physical_devices = tf.config.experimental.list_physical_devices("CPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0], [
        tf.config.experimental.VirtualDeviceConfiguration(),
        tf.config.experimental.VirtualDeviceConfiguration()
    ]
)

# tf.random.Generator 클래스를 통한 직접 객체 생성
g1_basic = tf.random.Generator.from_seed(1)  # 0 이상의 정수인 시드로부터 생성
g1_AlgorithmSelect = tf.random.Generator.from_seed(1, alg='philox')  # 생성기가 사용할 RNG 알고리즘 전달
print(g1_basic.normal(shape=[2, 3], dtype=tf.float64))  # dtype도 설정 가능
print(g1_AlgorithmSelect.normal(shape=[2, 3]))

# tf.random.get_global_generator() 를 호출하여 사용
g2 = tf.random.get_global_generator()
print(g2.normal(shape=[2, 3]))
