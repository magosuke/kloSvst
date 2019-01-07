# loading data
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 前処理
from keras.utilsnp_utils import to_categorical

# 画像を1次元化
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 画素を0~1の範囲に変換(正規化)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 正解ラベルをone-hot-encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#################################################################
# モデルを構築
#################################################################
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#################################################################
# モデルにデータを学習
#################################################################
model.fit(x_train, y_train,
    batch_size=100,
    epochs=12,
    verbose=1)      

#################################################################
# モデルを評価
#################################################################
score = model.evaluate(x_test, y_test)
print(score[0])
print(score[1])