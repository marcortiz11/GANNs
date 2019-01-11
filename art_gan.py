import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import os
import pickle as pkl

# Parameters
DATASET_PATH = '../Impressionism/Impressionism_32'
num_steps = 2000000
N = 20
zdim = 100
batch_size = 128
image_size = [32, 32]
logdir = "../tensorlogs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
saved_data = '../impressionism32.pkl'


# We define the generator
def generator(inp, z_dim):
    w1 = tf.get_variable('g_w1', [z_dim, 512 * 4 * 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(inp, w1)
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
    g1 = tf.nn.leaky_relu(g1, 0.2)
    g1 = tf.reshape(g1, [-1, 512, 4, 4])

    g2 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g1, [0, 2, 3, 1]), [8, 8]), [0, 1, 2, 3])
    w2 = tf.get_variable('g_w2', [3, 3, g2.get_shape()[3], 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g2, w2, strides=[1, 1, 1, 1], padding='SAME')
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
    g2 = tf.nn.leaky_relu(g2, 0.2)

    g3 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g2, [0, 2, 3, 1]), [16, 16]), [0, 1, 2, 3])
    w3 = tf.get_variable('g_w3', [3, 3, g3.get_shape()[3], 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g3, w3, strides=[1, 1, 1, 1], padding='SAME')
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
    g3 = tf.nn.leaky_relu(g3, 0.2)

    g4 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g3, [0, 2, 3, 1]), [32, 32]), [0, 1, 2, 3])
    w4 = tf.get_variable('g_w4', [3, 3, g4.get_shape()[3], 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g4, w4, strides=[1, 1, 1, 1], padding='SAME')
    g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, scope='g_b4')
    g4 = tf.nn.leaky_relu(g4, 0.2)

    # g5 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g4, [0, 2, 3, 1]), [64, 64]), [0, 1, 2, 3])
    # w5 = tf.get_variable('g_w5', [3, 3, g5.get_shape()[3], 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    # g5 = tf.nn.conv2d(g5, w5, strides=[1, 1, 1, 1], padding='SAME')
    # g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, scope='g_b5')
    # g5 = tf.nn.leaky_relu(g5, 0.2)

    w6 = tf.get_variable('g_w6', [3, 3, g4.get_shape()[3], 3], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g6 = tf.nn.conv2d(g4, w6, strides=[1, 1, 1, 1], padding='SAME')
    g6 = tf.nn.tanh(g6)

    print('generated image shape: ', g6.get_shape())
    return g6


# Discriminator
def discriminator(input):
    w1 = tf.get_variable('d_w1', [3, 3, input.get_shape()[3], 64])
    d1 = tf.nn.conv2d(input, w1, strides=[1, 2, 2, 1], padding='SAME')
    d1 = tf.nn.leaky_relu(d1, 0.2)
    print(d1.get_shape())

    w2 = tf.get_variable('d_w2', [3, 3, d1.get_shape()[3], 128])
    d2 = tf.nn.conv2d(d1, w2, strides=[1, 2, 2, 1], padding='SAME')
    d2 = tf.contrib.layers.batch_norm(d2, epsilon=1e-5, scope='d_b1')
    d2 = tf.nn.leaky_relu(d2, 0.2)
    print(d2.get_shape())

    w3 = tf.get_variable('d_w3', [3, 3, d2.get_shape()[3], 256])
    d3 = tf.nn.conv2d(d2, w3, strides=[1, 2, 2, 1], padding='SAME')
    d3 = tf.contrib.layers.batch_norm(d3, epsilon=1e-5, scope='d_b2')
    d3 = tf.nn.leaky_relu(d3, 0.2)
    print(d3.get_shape())

    w4 = tf.get_variable('d_w4', [3, 3, d3.get_shape()[3], 512])
    d4 = tf.nn.conv2d(d3, w4, strides=[1, 2, 2, 1], padding='SAME')
    d4 = tf.contrib.layers.batch_norm(d4, epsilon=1e-5, scope='d_b3')
    d4 = tf.nn.leaky_relu(d4, 0.2)
    print(d4.get_shape())

    w5 = tf.get_variable('d_w5', [3, 3, d4.get_shape()[3], 512])
    d5 = tf.nn.conv2d(d4, w5, strides=[1, 2, 2, 1], padding='SAME')
    d5 = tf.contrib.layers.batch_norm(d5, epsilon=1e-5, scope='d_b4')
    d5 = tf.nn.leaky_relu(d5, 0.2)
    d5 = tf.nn.avg_pool(d5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(d5.get_shape())

    flat = tf.reshape(d5, [-1, np.prod(d5.get_shape().as_list()[1:])])

    w6 = tf.get_variable('d_w6', [d5.get_shape()[-1], 1], initializer=tf.random_normal_initializer(stddev=0.02))
    d6 = tf.matmul(flat, w6)
    return d6


with tf.variable_scope('g'):
    #z_placeholder = tf.placeholder(tf.float32, [None, zdim])
    z = tf.random_uniform([batch_size, zdim], -1, 1)
    G = generator(z, zdim)

with tf.variable_scope('d') as scope:
    x = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 3])
    D1 = discriminator(x)
    scope.reuse_variables()
    D2 = discriminator(G)


# TODO: change optimizer. Adam?
def optimizer(loss, var_list):
    initial_learning_rate = 0.005
    decay = 0.95
    num_decay_steps = 1000
    batch = tf.Variable(N)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    opt = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return opt


def batch_generator(x_data):
    while True:
        for i in range(len(x_data)):
            x_batch = []
            for j in range(batch_size):
                tmp = Image.open(x_data[i])
                tmp.load()
                x_batch.append(np.asarray(tmp))
                tmp.close()
            yield x_batch


# TODO: may need to redo losses
# FIXME: sigmoid or softmax here?
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1, labels=tf.ones_like(D1)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.ones_like(D2)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.ones_like(D2)))

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('d/')]
g_params = [v for v in vars if v.name.startswith('g/')]
print(d_params)
print(g_params)

d_opt_real = optimizer(d_loss_real, d_params)
d_opt_fake = optimizer(d_loss_fake, d_params)
g_opt = optimizer(g_loss, g_params)

d_loss_record = np.zeros(num_steps)
g_loss_record = np.zeros(num_steps)

sess = tf.Session()

# Little summary
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Disciminator_loss', d_loss)

generated_images_tensor = generator(z, zdim)
tf.summary.image('Generated_images', generated_images_tensor, 5)
sm = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())


if os.path.isfile(saved_data):
    data = pkl.load(open(saved_data, 'rb'))
else:
    data = []
    for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
        for file in filenames:
            data.append(os.path.join(dirpath, file))
    pkl.dump(data, open(saved_data, 'wb'))

batch_gen = batch_generator(data)
saver = tf.train.Saver()

for step in range(num_steps):
    real_batch = next(batch_gen)
    #z_batch = tf.random_uniform([batch_size, zdim], -1, 1)

    # update discriminator
    _, _, d_lossR, d_lossF = sess.run([d_opt_real, d_opt_fake, d_loss_real, d_loss_fake],
                                      feed_dict={x: real_batch})

    #z_batch = tf.random_uniform([batch_size, zdim], -1, 1)
    _ = sess.run(g_opt)

    if step % 100 == 0:
        #z_batch = tf.random_uniform([batch_size, zdim], -1, 1)
        print(step, 'Completed')
        summary = sess.run(sm, feed_dict={x: real_batch})
        summary_writer.add_summary(summary, step)
        saver.save(sess, "/tmp/model.ckpt")
