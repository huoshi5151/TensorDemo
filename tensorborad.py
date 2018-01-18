import tensorflow as tf
import numpy as np

## make up some data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

def add_layer(inputs ,
              in_size,
              out_size,n_layer,
              activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
         with tf.name_scope('weights'):
              Weights= tf.Variable(tf.random_normal([in_size, out_size]),name='W')
              # tf.histogram_summary(layer_name+'/weights',Weights)
              tf.summary.histogram(layer_name + '/weights', Weights) # tensorflow >= 0.12

         with tf.name_scope('biases'):
              biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
              # tf.histogram_summary(layer_name+'/biase',biases)
              tf.summary.histogram(layer_name + '/biases', biases)  # Tensorflow >= 0.12

         with tf.name_scope('Wx_plus_b'):
              Wx_plus_b = tf.add(tf.matmul(inputs,Weights), biases)

         if activation_function is None:
            outputs=Wx_plus_b
         else:
            outputs= activation_function(Wx_plus_b)

         # tf.histogram_summary(layer_name+'/outputs',outputs)
         tf.summary.histogram(layer_name + '/outputs', outputs) # Tensorflow >= 0.12

    return outputs


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)  # tensorflow >= 0.12

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

merged = tf.summary.merge_all() # tensorflow >= 0.12
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())  # 替换成这样就好

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs

for i in range(1000):
   sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
   if i%50 == 0:
      rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
      writer.add_summary(rs, i)
