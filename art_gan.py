import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import PIL
import os
import pickle as pkl
import matplotlib.pyplot as plt

# Parameters
image_size = [32, 32]  # Just need to change this and all the model adapts
DATASET_PATH = '../Impressionism/Impressionism_' + str(image_size[0])
MODELS_PATH = '/home/magi/mai/ci/models'
MODELS_PATH = os.path.join(MODELS_PATH, 'models' + str(image_size[0]) + '/')
print(MODELS_PATH)
num_steps = 2000000
zdim = 100
batch_size = 64
logdir = "../tensorlogs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
saved_data = '../impressionism' + str(image_size[0]) + '.pkl'
learning_rate_method = 'constant'
resume_training = False


def generator(inp, img_size, reuse=False):
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

        if img_size == 32:
            logits = tf.layers.conv2d_transpose(g4, 3, 1, 1, 'same')
            out = tf.tanh(logits)

        if img_size == 64 or img_size == 128:
            # 64x64x64
            g5 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g4, [0, 2, 3, 1]), [64, 64]), [0, 1, 2, 3])
            w5 = tf.get_variable('g_w5', [3, 3, g5.get_shape()[3], 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            g5 = tf.nn.conv2d(g5, w5, strides=[1, 1, 1, 1], padding='SAME')
            g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, scope='g_b5')
            g5 = tf.nn.relu(g5)

            if img_size == 64:
                logits = tf.layers.conv2d_transpose(g5, 3, 1, 1, 'same')
                out = tf.tanh(logits)
            elif img_size == 128:
                # 128x128x32
                g6 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g5, [0, 2, 3, 1]), [128, 128]), [0, 1, 2, 3])
                w6 = tf.get_variable('g_w6', [3, 3, g6.get_shape()[3], 32],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
                g6 = tf.nn.conv2d(g6, w6, strides=[1, 1, 1, 1], padding='SAME')
                g6 = tf.contrib.layers.batch_norm(g6, epsilon=1e-5, scope='g_b6')
                g6 = tf.nn.relu(g6)

                logits = tf.layers.conv2d_transpose(g6, 3, 1, 1, 'same')
                out = tf.tanh(logits)

    return out


def discriminator(input, img_size, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is HeightxWidthx3

        # Add noise as a mode collapse prevention
        input = input + tf.random_normal(tf.shape(input), stddev=0.05)
        print('input size: ', input.get_shape())

        # 16x16x32 || 32x32x32 || 64x64x32
        w1 = tf.get_variable('d_w1', [3, 3, input.get_shape()[3], 32])
        d1 = tf.nn.conv2d(input, w1, strides=[1, 2, 2, 1], padding='SAME')
        d1 = tf.nn.leaky_relu(d1, 0.2)

        # 8x8x64 || 16x16x64 || 32x32x64
        d2 = tf.layers.conv2d(d1, 64, 3, 2, 'same', use_bias=False)
        d2 = tf.layers.batch_normalization(d2)
        d2 = tf.nn.leaky_relu(d2, 0.2)

        # 4x4x128 || 8x8x128 || 16x16x128
        d3 = tf.layers.conv2d(d2, 128, 3, 2, 'same', use_bias=False)
        d3 = tf.layers.batch_normalization(d3)
        d3 = tf.nn.leaky_relu(d3, 0.2)

        # 2x2x256 || 4x4x256 || 8x8x256
        d4 = tf.layers.conv2d(d3, 256, 3, 2, 'same', use_bias=False)
        d4 = tf.layers.batch_normalization(d4)
        d4 = tf.nn.leaky_relu(d4, 0.2)
        d4 = tf.nn.dropout(d4, 0.5)

        # 1x1x512 || 2x2x512 || 4x4x512
        d5 = tf.layers.conv2d(d4, 512, 3, 2, 'same', use_bias=False)
        d5 = tf.layers.batch_normalization(d5)
        d5 = tf.nn.leaky_relu(d5, 0.2)
        d5 = tf.nn.dropout(d5, 0.5)

        if img_size == 32:
            flat = tf.reshape(d5, [-1, np.prod(d5.get_shape().as_list()[1:])])
            logits = tf.layers.dense(flat, 1)
            out = tf.sigmoid(logits)
        else:
            # 1x1x1024 || 2x2x1024
            d6 = tf.layers.conv2d(d5, 1024, 3, 2, 'same', use_bias=False)
            d6 = tf.layers.batch_normalization(d6)
            d6 = tf.nn.leaky_relu(d6, 0.2)

            if img_size == 64:
                d6 = tf.nn.dropout(d6, 0.5)
                flat = tf.reshape(d6, [-1, np.prod(d6.get_shape().as_list()[1:])])
                logits = tf.layers.dense(flat, 1)
                out = tf.sigmoid(logits)

        if img_size == 128:
            # 1x1x2048
            d7 = tf.layers.conv2d(d6, 2048, 3, 2, 'same', use_bias=False)
            d7 = tf.layers.batch_normalization(d7)
            d7 = tf.nn.leaky_relu(d7, 0.2)

            flat = tf.reshape(d7, [-1, np.prod(d7.get_shape().as_list()[1:])])
            logits = tf.layers.dense(flat, 1)
            out = tf.sigmoid(logits)

        return out, logits


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


with tf.Session() as sess:

    z_placeholder = tf.placeholder(tf.float32, [None, zdim])
    G = generator(z_placeholder, img_size=image_size[0])

    x = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 3])
    D1, D1_logits = discriminator(x, img_size=image_size[0])
    D2, D2_logits = discriminator(G, img_size=image_size[0], reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logits, labels=tf.constant(0.9)*tf.ones_like(D1)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.zeros_like(D2)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.ones_like(D2)))

    vars = tf.trainable_variables()
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

    if resume_training:
        last_checkpoint = tf.train.latest_checkpoint(MODELS_PATH)
        print('Loading model from: ', last_checkpoint)
        saver.restore(sess, last_checkpoint)

    generated_images_tensor = generator(z_placeholder, img_size=image_size[0], reuse=True)
    tf.summary.image('Generated_images', generated_images_tensor, 10)
    tf.summary.image('Real images', x, 2)
    sm = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)

    if not resume_training:
        sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.local_variables_initializer())
    print('Computing gan')
    step = 0
    while True:
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
            if step % 100 == 0:
                save_path = saver.save(sess, MODELS_PATH + "drop_out_model-checkpoint_step_{}_size{}.ckpt".
                                       format(step, image_size[0]), global_step=step)
