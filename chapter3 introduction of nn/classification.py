# 这是一个关于电影影评的二分类代码问题
# 3.1加载IMDB数据集
from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)     # 数据集与测试集的加载，num_words的意思是保留训练数据中前10000个最常出现的单词
print(test_data.shape)
print(train_data[0])
'''
print(train_data.shape)
print(test_data.shape)    
'''
word_index = imdb.get_word_index()     # word_index是一个将单词映射为整数索引的字典
# print(word_index)    # 单词映射为整数的字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])   # 键值颠倒，将整数索引映射为单词
decode_review = ' '.join(reverse_word_index.get(i - 3, '?') for i in train_data[0])  # 评论的解码，索引-3是为了将0,1,2的其它索引去掉  .join方法转换为一个字符串类型的输出
print(decode_review)

# 3.2将整数序列编码为二进制矩阵

print(train_data[0])

def vectorize_sequnces(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))    # 创建一个零维的矩阵
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1      # 将results[i,sequences]的指定索引设为1,注意这是一个刚刚创建的0维矩阵
    return results


x_train = vectorize_sequnces(train_data)   # 将训练数据向量化
x_test = vectorize_sequnces(test_data)      # 将测试数据向量化

# 将标签向量化 , 矩阵化为32的浮点数。
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(y_train)
#print(x_train[0])

# 模型定义
from keras import models
from keras import  layers

# 模型定义
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# 编译模型
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 配置优化器
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 使用自定义的损失和指标
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


# validation set 的部分保留  将原始训练数据的10000份留出作为保留
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 模型的训练
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=5,
                    batch_size=512,
                    validation_data=(x_val, y_val))


import matplotlib.pyplot as plt

history_dict = history.history
loss_value = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_value)+1)

plt.plot(epochs, loss_value, 'bo', label='Training loss')       # 其中bo表示蓝点
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') # b 表示的是蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# 绘制训练精度与测试精度
plt.clf()         # 清空图像
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()

# 从上述图像可以看到，由于训练集上的精度越来越高，验证集上不是，这是一个过拟合的现象
'''
# 3.11从头开始训练一个模型
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crosspentroy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4,batch_size=512)   # 训练4轮即可
results = model.evaluate(x_test, y_test)

 
history_dict = history.history
loss_value = history_dict['loss']
val_loss_values = history_dict['val_loss']

plt.clf()         # 清空图像
acc = history_dict['acc']
val_acc = history_dict['val_acc']



plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()
'''