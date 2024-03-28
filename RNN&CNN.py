import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import TensorBoard
from keras.src.layers import Dense, LSTM, Dropout, Flatten, Conv1D, BatchNormalization, Activation, MaxPooling1D
from keras.src.regularizers import l2
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import plotAUCforMulticlass

dataSet = pd.read_csv('Train.csv')
xData = dataSet.iloc[:, 1:-1]
yData = dataSet.iloc[:, -1]

scalar = preprocessing.StandardScaler()
xData = scalar.fit_transform(xData)

yData = np.array(yData).reshape([-1, 1])            # 转化为1列
Encoder = preprocessing.OneHotEncoder()             # 独热编码
Encoder.fit(yData)
yData = Encoder.transform(yData).toarray()
yData = np.asarray(yData, dtype=np.int32)

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

number = xTrain.shape[1]
batchSize = 8
epochs = 20         # 循环周期
numClasses = 10     # 分10类

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4), input_shape=(xTrain.shape[1], 1)))
print(model.summary())
model.add(Conv1D(filters=32))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(16, return_sequences=True))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(numClasses, activation='softmax', kernel_regularizer=l2(1e-4)))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
tb_cb = TensorBoard(log_dir='logs')     # 查看训练情况
H = model.fit(x=xTrain, y=yTrain, batch_size=batchSize, epochs=epochs, verbose=1, shuffle=True, callbacks=[tb_cb])      # 开始模型训练

score = model.evaluate(x=xTest, y=yTest, verbose=0)     # 训练结果
print("测试集上的损失：", score[0])
print("测试集上的准确度：", score[1])

N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["accuracy"], label="train_accuracy")
plt.title("Training loss and Accuracy(Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

yPredict = model.predict(xTest)
plotAUCforMulticlass.plot_multiclass_AUC(numClasses, yTest, yPredict)