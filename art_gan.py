import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import PIL
import os
import pickle as pkl
import matplotlib.pyplot as plt

# Parameters
DATASET_PATH = '../Impressionism/Impressionism_64'
num_steps = 2000000
zdim = 100
batch_size = 64
image_size = [64, 64]
logdir = "../tensorlogs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
saved_data = '../impressionism64.pkl'
learning_rate_method = 'constant'


# TODO: generator doesn't create good images. Discriminator always detect the fake images
# The problem may be that the generator creates too similar images whatever is the random values 

# We define the generator
def generator(inp, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 4x4x1024
        g1 = tf.layers.dense(inp, 4 * 4 * 1024, use_bias=False)
        g1 = tf.reshape(g1, [-1, 4, 4, 1024])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
        g1 = tf.nn.relu(g1)

        # 8x8x512
        g2 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g1, [0, 2, 3, 1]), [8, 8]), [0, 1, 2, 3])
        w2 = tf.get_variable('g_w2', [3, 3, g2.get_shape()[3], 512],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        g2 = tf.nn.conv2d(g2, w2, strides=[1, 1, 1, 1], padding='SAME')
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
        g2 = tf.nn.relu(g2)

        # 16x16x256
        g3 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g2, [0, 2, 3, 1]), [16, 16]), [0, 1, 2, 3])
        w3 = tf.get_variable('g_w3', [3, 3, g3.get_shape()[3], 256],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = tf.nn.conv2d(g3, w3, strides=[1, 1, 1, 1], padding='SAME')
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
        g3 = tf.nn.relu(g3)

        # 32x32x128
        g4 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g3, [0, 2, 3, 1]), [32, 32]), [0, 1, 2, 3])
        w4 = tf.get_variable('g_w4', [3, 3, g4.get_shape()[3], 128],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        g4 = tf.nn.conv2d(g4, w4, strides=[1, 1, 1, 1], padding='SAME')
        g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, scope='g_b4')
        g4 = tf.nn.relu(g4)

        # # 64x64x64
        g5 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g4, [0, 2, 3, 1]), [64, 64]), [0, 1, 2, 3])
        w5 = tf.get_variable('g_w5', [3, 3, g5.get_shape()[3], 64],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        g5 = tf.nn.conv2d(g5, w5, strides=[1, 1, 1, 1], padding='SAME')
        g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, scope='g_b5')
        g5 = tf.nn.relu(g5)

        # 64x64x3
        logits = tf.layers.conv2d_transpose(g5, 3, 1, 1, 'same')
        out = tf.tanh(logits)
    return out


def discriminator(input, reuse=False, alpha=0.2, keep_prob=0.5):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 32x32x3

        # Add noise as a mode collapse prevention
        input = input + tf.random_normal(tf.shape(input), stddev=0.05)

        # 16x16x32
        w1 = tf.get_variable('d_w1', [3, 3, input.get_shape()[3], 32])
        d1 = tf.nn.conv2d(input, w1, strides=[1, 2, 2, 1], padding='SAME')
        d1 = tf.nn.leaky_relu(d1, 0.2)

        # 8x8x64
        d2 = tf.layers.conv2d(d1, 64, 3, 2, 'same', use_bias=False)
        d2 = tf.layers.batch_normalization(d2)
        d2 = tf.nn.leaky_relu(d2, 0.2)
        #d2 = tf.layers.dropout(d2, keep_prob)

        # 4x4x128
        d3 = tf.layers.conv2d(d2, 128, 3, 2, 'same', use_bias=False)
        d3 = tf.layers.batch_normalization(d3)
        d3 = tf.nn.leaky_relu(d3, 0.2)
        #d3 = tf.layers.dropout(d3, keep_prob)

        # 2x2x256
        d4 = tf.layers.conv2d(d3, 256, 3, 2, 'same', use_bias=False)
        d4 = tf.layers.batch_normalization(d4)
        d4 = tf.nn.leaky_relu(d4, 0.2)
        #d4 = tf.layers.dropout(d4, keep_prob)

        # 1x1x512
        d5 = tf.layers.conv2d(d4, 512, 3, 2, 'same', use_bias=False)
        d5 = tf.layers.batch_normalization(d5)
        d5 = tf.nn.leaky_relu(d5, 0.2)
        #d5 = tf.layers.dropout(d5, keep_prob)

        flat = tf.reshape(d5, [-1, np.prod(d5.get_shape().as_list()[1:])])
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits


z_placeholder = tf.placeholder(tf.float32, [None, zdim])
G = generator(z_placeholder)

x = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 3])
D1, D1_logits = discriminator(x)
D2, D2_logits = discriminator(G, reuse=True)


def optimizer(loss, var_list, gen=False, method='adaptive'):
    batch = tf.Variable(batch_size)
    if method == 'adaptive':
        initial_learning_rate = 0.00025
        if gen:
            initial_learning_rate = 0.00008
        decay = 0.95
        num_decay_steps = 200
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            batch,
            num_decay_steps,
            decay,
            staircase=True
        )
    else:
        learning_rate = 0.0003
        if gen:
            # Generator learning rate
            learning_rate = 0.0002
    opt = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return opt


def batch_generator(x_data):
    while True:
        x_batch = []
        for j in range(batch_size):
            index = np.random.choice(len(x_data), 1)
            tmp = Image.open(x_data[index[0]])
            x_batch.append(np.asarray(tmp))
            tmp.close()
        yield np.array(x_batch) / 255 - 0.5


# FIXME: the losses may be the problem
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logits, labels=tf.constant(0.9)*tf.ones_like(D1)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.zeros_like(D2)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.ones_like(D2)))

vars = tf.trainable_variables()
print(vars)
d_params = [var for var in vars if var.name.startswith('discriminator')]
g_params = [var for var in vars if var.name.startswith('generator')]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_opt = optimizer(d_loss, d_params, method=learning_rate_method)
    g_opt = optimizer(g_loss, g_params, gen=True, method=learning_rate_method)

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

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Disciminator_loss', d_loss)

with tf.Session() as sess:

    generated_images_tensor = generator(z_placeholder, reuse=True)
    tf.summary.image('Generated_images', generated_images_tensor, 10)
    tf.summary.image('Real images', x, 2)
    sm = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)

    sess.run(tf.global_variables_initializer())
    print('Computing gan')
    step = 0
    epoch = 0
    while True:  # EPOCHS
        for real_batch in batch_gen:
            # real_batch = next(batch_gen)
            step += 1
            z_batch = np.random.uniform(-1, 1, size=[batch_size, zdim])

            # update discriminator
            _, dloss = sess.run([d_opt, d_loss], feed_dict={x: real_batch, z_placeholder: z_batch})

            # update generator
            _, gloss = sess.run([g_opt, g_loss], feed_dict={z_placeholder: z_batch})

            if step % 100 == 0:
                print(step, 'Completed')
                summary = sess.run(sm, feed_dict={x: real_batch, z_placeholder: z_batch})
                summary_writer.add_summary(summary, step)
            if step % 2000 == 0:
                save_path = saver.save(sess, "../models/model-checkpoint_step%s.ckpt" % step)
        epoch += 1
        print('%i epoch finished' % epoch)
        save_path = saver.save(sess, "../models/model-checkpoint_epoch%s.ckpt" % epoch)
        print('Model saved in path: %s' % save_path)
