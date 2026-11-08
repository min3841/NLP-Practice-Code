from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split



values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1],
['서류는 내일까지 보내주시면 됩니다.', 1]]
columns = ['메일 본문', '스팸 메일 유무']

data = DataFrame(values, columns= columns)


x = data['메일 본문']
y = data['스팸 메일 유무']

num_of_train = int(len(x) * 0.6)
num_of_test = int(len(x) - num_of_train)



X_test = x[num_of_train : ]
y_test = y[num_of_train:]
X_train = x[:num_of_train]
y_train = y[:num_of_train]

print('X 테스트 데이터 :')
print(list(X_test))
print('y 테스트 데이터 :')
print(list(y_test))
print('X 학습 데이터 :')
print(list(X_train))
print('y 학습 데이터 :')
print(list(y_train))



# 데이터 분리

""" 

data = [['안녕','한국어'],['hello','영어'],['뭐 먹을까','한국어']]

x,y = zip(*data)

print(f"x:{x}\ny:{y}")  """


"""
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']


df = pd.DataFrame(values, columns= columns)


X = df['메일 본문']

Y = df['스팸 메일 유무']

print(f"x:{X.to_list()}\ny:{Y.to_list()}") """



"""
np_array = np.arange(0,16).reshape((4,4))

X = np_array[:, :3]

Y = np_array[:, 3]

print(f"x:{X}\ny:{Y}") """



# 학습, 테스트 데이터 분리
"""
X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)

print(f"X_train:{X_train}\nX_test:{X_test}")
print(f"y_train:{y_train}\ny_test:{y_test}") """