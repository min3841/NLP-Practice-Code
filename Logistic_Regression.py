import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

x = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])
y = np.array([0, 0, 0, 1, 1, 1])

model = Sequential()

model.add(Dense(1,input_dim=2, activation='sigmoid'))

sgd = optimizers.SGD(learning_rate=0.0001)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(x,y,epochs=500)
y_pred = model.predict(x)
# 실제 데이터 시각화
plt.scatter(x[y==0, 0], x[y==0, 1], color='blue', label= 'Class 0')
plt.scatter(x[y==1, 0], x[y==1, 1], color='red', label= 'Class 1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression')
plt.legend()
plt.grid()
plt.savefig('regression.png', dpi=150, bbox_inches='tight')
plt.show()

print(model.predict(np.array([[-1, 3], [5, -3]])))

