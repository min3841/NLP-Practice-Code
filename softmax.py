import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/06.%20Machine%20Learning/dataset/Iris.csv", filename="dataset/Iris.csv")

data = pd.read_csv('dataset/Iris.csv', encoding='utf-8-sig')

data['Species'] = data['Species'].replace(['Iris-virginica','Iris-setosa','Iris-versicolor'], [0,1,2])

data_X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
data_Y = data['Species'].values


# 전체 데이터에서 테스트 데이터 20% 분리
X_train_val, X_test, y_train_val, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=1, stratify=data_Y)

# 남은 데이터에서 검증 데이터 25% 분리
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=1, stratify=y_train_val)

print(f"학습x 데이터:{len(X_train)}\n")
print(f"학습y 데이터:{len(y_train)}\n")
print(f"검증x 데이터:{len(X_val)}\n")
print(f"검증y 데이터:{len(y_val)}\n")
print(f"테스트x 데이터:{len(X_test)}\n")
print(f"테스트y 데이터:{len(y_test)}\n")

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Dense(3, input_dim=4, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

Early_Stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_val, y_val), callbacks=[Early_Stopping])

epochs = range(1,len(history.history['accuracy']) + 1)
plt.plot(epochs,history.history['loss'] )
plt.plot(epochs,history.history['val_loss'] )
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig("softmax", dpi=150, bbox_inches='tight')

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

