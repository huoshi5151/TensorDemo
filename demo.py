from __future__ import print_function
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np


### 从文件获取数据
my_matrix = np.loadtxt(open("data2.csv", "rb"), dtype=np.float, delimiter=",", skiprows=1)

tezheng = my_matrix[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
money = my_matrix[:,0]
print(money)
X_train = Normalizer().fit_transform(tezheng)
y_train = money.reshape((-1,1))

print(y_train)
X_test = X_train[:1]

### 从文件获取测试数据
my_matrix2 = np.loadtxt(open("data3.csv", "rb"), dtype=np.float, delimiter=",", skiprows=1)

tezheng2 = my_matrix2[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
money2 = my_matrix2[:,0]

X_test2 = Normalizer().fit_transform(tezheng2)
y_test2 = money2.reshape((-1,1))


#### 开始进行图的构建
inputX = tf.placeholder(shape=[None, X_train.shape[1]], dtype=tf.float32, name="inputX")
y_true = tf.placeholder(shape=[None,1], dtype=tf.float32, name="y_true")

keep_prob_s = tf.placeholder(dtype=tf.float32, name="keep_prob")

Weights1 = tf.Variable(tf.random_normal(shape=[52, 10]), name="weights1")
biases1 = tf.Variable(tf.zeros(shape=[1, 10]) + 0.1, name="biases1")

Wx_plus_b1 = tf.matmul(inputX, Weights1) + biases1
Wx_plus_b1 = tf.nn.dropout(Wx_plus_b1, keep_prob=keep_prob_s)

l1 = tf.nn.sigmoid(Wx_plus_b1, name="l1")

Weights2 = tf.Variable(tf.random_normal(shape=[10, 1]), name="weights2")
biases2 = tf.Variable(tf.zeros(shape=[1, 1]) + 0.1, name="biases2")

Wx_plus_b2 = tf.matmul(l1, Weights2)
prediction = tf.add(Wx_plus_b2, biases2, name="pred")

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - prediction), reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


#### draw pics
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(50), y_train[0:50], 'b')  #展示前50个数据
ax.set_ylim([0, 30])
plt.ion()
plt.show()


### 开始执行
with tf.Session() as sess:

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)          ### 保存

    init = tf.global_variables_initializer()
    sess.run(init)
    feed_dict_train = {inputX: X_train, y_true: y_train, keep_prob_s: 1}

    for i in range(40000):
        _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict_train)
        if i % 1000 == 0:
            print("步数:%d\tloss:%.5f" % (i, _loss))
            pred_feed_dict = {inputX: X_train, keep_prob_s: 1}
            pred = sess.run(prediction, feed_dict=pred_feed_dict)
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            lines = ax.plot(range(50), pred[0:50], 'r--')
            plt.pause(1)

    saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=40000)  # 保存模型

    ### 预测
    # pred_feed_dict = {inputX: X_test, keep_prob_s: 1}
    # pred_feed_dict2 = {inputX: X_test2, keep_prob_s: 1}
    # pred1 = sess.run(prediction, feed_dict=pred_feed_dict)
    # pred2 = sess.run(prediction, feed_dict=pred_feed_dict2)
    #
    # print("====")
    # print(pred1)
    # print("====")
    # print(pred2)