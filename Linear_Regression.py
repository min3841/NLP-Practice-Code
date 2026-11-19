from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
y = np.array([11, 22, 33, 44, 53, 66, 77, 87, 95], dtype=np.float32)

x = x.reshape(-1, 1)

model = Sequential()

model.add(Dense(1, input_dim=1, activation='linear'))

sgd = optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

model.fit(x, y, epochs=300)
y_pred = model.predict(x)
plt.plot(x.flatten(), y_pred.flatten(), 'b-', label='Predicted')
plt.scatter(x.flatten(), y, color='black', label='Actual')

plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.title('Linear Regression')
plt.legend()
plt.grid()
plt.savefig('regression.png', dpi=150, bbox_inches='tight')

print(model.predict(np.array([[9.5]], dtype=np.float32)))





"""
import tensorflow as tf

x = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적

w = tf.Variable(7.0)
b = tf.Variable(1.0)

@tf.function #그래프 연산을 가능하게 하는 데코레이터
def hypothesis(x):

    return w*x + b

@tf.function
def mse_loss(y_pred,y):

    return tf.reduce_mean(tf.square(y_pred-y))

optimizer = tf.optimizers.SGD(0.01)

for i in range(301):

    with tf.GradientTape() as tape:

        y_pred = hypothesis(x)

        cost = mse_loss(y_pred,y)

    gradients = tape.gradient(cost,[w,b])

    optimizer.apply_gradients(zip(gradients,[w,b]))

    if i % 10 == 0:

        print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))

    x_test = [3.5, 5, 5.5, 6]

    print(hypothesis(x_test).numpy())

"""