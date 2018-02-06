import tensorflow as tf
from sklearn.preprocessing import Normalizer
import numpy as np

### 从文件获取测试数据
my_matrix2 = np.loadtxt(open("data3.csv", "rb"), dtype=np.float, delimiter=",", skiprows=1)

tezheng2 = my_matrix2[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]

X = Normalizer().fit_transform(tezheng2)

with tf.Session() as sess:

    # 拿到 图的元信息 saver
    saver = tf.train.import_meta_graph(meta_graph_or_file="nn_boston_model/nn_boston.model-40000.meta")          

    # 这个也很重要，虽然还不知道是做什么的
    model_file = tf.train.latest_checkpoint(checkpoint_dir="nn_boston_model")

    # 执行恢复
    saver.restore(sess=sess, save_path=model_file)

    # 此处得到的是图结构
    graph = tf.get_default_graph()

    # get placeholder from graph  拿到两个需要输入的数据节点，预测时输入
    inputX = graph.get_tensor_by_name("inputX:0")
    keep_prob_s = graph.get_tensor_by_name("keep_prob:0")

    # get operation from graph   拿到能输出结果的数据节点，预测时执行到这里，拿到预测结果值
    prediction = graph.get_tensor_by_name("pred:0")

    # 开始预测, 还是不用输入y , 丢弃我一直写的1，不丢弃
    feed_dict = {inputX: X, keep_prob_s: 1}
    y_pred = sess.run(prediction, feed_dict=feed_dict)

    # 得到预测结果
    print(y_pred)
