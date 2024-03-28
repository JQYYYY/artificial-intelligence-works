import warnings
from collections import Counter
from itertools import cycle

import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.engine.saving import model_from_json
from keras.layers import MaxPooling1D, Conv1D, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, \
    precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')
numClass = 3
classLabel = ['high', 'medium', 'low']
sheet0 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=0, usecols="C:L")       # sheet_name=None读取全部sheet
sheet0.loc[:, 'label'] = 0          # 列增加label
sheet0.iloc[:, 0:-1] = np.sqrt(sheet0.iloc[:, 0:-1])
sheet1 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=1, usecols="C:L")
sheet1.loc[:, 'label'] = 1
sheet1.iloc[:, 0:-1] = np.sqrt(sheet1.iloc[:, 0:-1])
sheet2 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=2, usecols="C:L")
sheet2.loc[:, 'label'] = 2
sheet2.iloc[:, 0:-1] = np.sqrt(sheet2.iloc[:, 0:-1])
sheets = pd.concat([sheet0, sheet1, sheet2])      # 按行合并三张表

x = sheets.iloc[:, 0:-1]
y = sheets.iloc[:, -1]             # 获得标签
y = np.array(y).reshape(-1, 1)
x = np.expand_dims(x, axis=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
y_temp = y_test
Encoder = preprocessing.OneHotEncoder()
y_test = Encoder.fit_transform(y_test).toarray()
y_test = np.asarray(y_test, dtype=np.int32)
y_train = Encoder.fit_transform(y_train).toarray()
y_train = np.asarray(y_train, dtype=np.int32)
# x_train, x_test = x_train[:, :, np.newaxis], x_test[:, :, np.newaxis]    # 增加通道数目

# print(x_train.shape)
# 搭建模型

def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(10, 1)))
    model.add(Conv1D(16, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation='tanh', padding='same'))
    model.add(Conv1D(32, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation='tanh', padding='same'))
    model.add(Conv1D(32, 3, activation='tanh', padding='same'))
    # model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


epochs = 20
# 训练分类器
# estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs, batch_size=1, verbose=1)
# tb_cb = TensorBoard(log_dir='logs')     # 查看训练情况
# H = estimator.fit(x_train, y_train, callbacks=[tb_cb])
# score = estimator.score(x_test, y_test)
# print(score)

font = {'family': 'Times New Roman'}
plt.rc('font', **font)

# N = np.arange(0, epochs)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["acc"], label="train_acc")
# plt.title("Training loss and Accuracy(Simple NN)")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()

# 将其模型转换为json
# model_json = estimator.model.to_json()
# with open(r"model.json",'w')as json_file:
#     json_file.write(model_json)# 权重不在json中,只保存网络结构
# estimator.model.save_weights('model.h5')

# 加载模型
json_file = open(r"model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 分类准确率
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f"loss:{score[0]}     accuracy:{score[1]}")
y_pred = loaded_model.predict(x_test)
predicted_labels = np.argmax(y_pred, axis=1)
# print(predicted_labels)

# 混淆矩阵
true_label = y_test.argmax(axis=-1)  # 将one-hot转化为label
conf_mat = confusion_matrix(y_true=true_label, y_pred=predicted_labels)
ConfusionMatrixDisplay(conf_mat).plot()
plt.show()

target_names = ['high', 'medium', 'low']
fpr, tpr, roc_auc = dict(), dict(), dict()
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
    color="deeppink",
    linestyle="-",
    linewidth=1,
)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

recall, precision, ap = dict(), dict(), dict()
recall["micro"], precision["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
ap["micro"] = average_precision_score(y_test, y_pred, average='micro')
plt.plot(recall["micro"], precision["micro"], label='micro-average PR curve(MAP={:.2f})'.format(ap["micro"]))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.legend()
plt.show()