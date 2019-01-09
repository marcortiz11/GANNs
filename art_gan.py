import tensorflow as tf
import numpy as np

# Parameters
DATASET_PATH = '../Impressionism'
num_steps = 1000
N = 20
zdim = 100
batch_size = 128

z = tf.random_uniform([batch_size, zdim], -1, 1)


# We define the generator
def generator(inp, z_dim):
    w1 = tf.get_variable('g_w1', [z_dim, 512 * 4 * 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(inp, w1)
    g1 = tf.reshape(g1, [-1, 512, 4, 4])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
    g1 = tf.nn.leaky_relu(g1, 0.2)

    w2 = tf.get_variable('g_w2', [3, 3, g1.get_shape()[1], 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g1, [0, 2, 3, 1]), [8, 8]), [0, 3, 1, 2])
    g2 = tf.nn.conv2d(g2, w2, strides=[1, 1, 1, 1])
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
    g2 = tf.nn.leaky_relu(g2, 0.2)

    w3 = tf.get_variable('g_w3', [3, 3, g2.get_shape()[1], 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g2, [0, 2, 3, 1]), [16, 16]), [0, 3, 1, 2])
    g3 = tf.nn.conv2d(g3, w3, strides=[1, 1, 1, 1])
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
    g3 = tf.nn.leaky_relu(g3, 0.2)

    w4 = tf.get_variable('g_w4', [3, 3, g3.get_shape()[1], 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g3, [0, 2, 3, 1]), [32, 32]), [0, 3, 1, 2])
    g4 = tf.nn.conv2d(g4, w4, strides=[1, 1, 1, 1])
    g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, scope='g_b4')
    g4 = tf.nn.leaky_relu(g4, 0.2)

    w5 = tf.get_variable('g_w5', [3, 3, g4.get_shape()[1], 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g5 = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(g4, [0, 2, 3, 1]), [64, 64]), [0, 3, 1, 2])
    g5 = tf.nn.conv2d(g5, w5, strides=[1, 1, 1, 1])
    g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, scope='g_b5')
    g5 = tf.nn.leaky_relu(g5, 0.2)

    w6 = tf.get_variable('g_w6', [3, 3, g5.get_shape()[1], 3], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g6 = tf.nn.conv2d(g5, w6, strides=[1, 1, 1, 1])
    g6 = tf.nn.tanh(g6)
    return g6


# Discriminator
def discriminator(input):
    w1 = tf.get_variable('d_w1', [3, 3, input.get_shape()[1], 128])
    d1 = tf.nn.conv2d(input, w1, strides=[1, 1, 2, 2])
    d1 = tf.nn.leaky_relu(d1, 0.2)

    w2 = tf.get_variable('d_w2', [3, 3, d1.get_shape()[1], 256])
    d2 = tf.nn.conv2d(d1, w2, strides=[1, 1, 2, 2])
    d2 = tf.contrib.layers.batch_norm(d2, epsilon=1e-5, scope='d_b1')
    d2 = tf.nn.leaky_relu(d2, 0.2)

    w3 = tf.get_variable('d_w3', [3, 3, d2.get_shape()[1], 512])
    d3 = tf.nn.conv2d(d2, w3, strides=[1, 1, 2, 2])
    d3 = tf.contrib.layers.batch_norm(d3, epsilon=1e-5, scope='d_b2')
    d3 = tf.nn.leaky_relu(d3, 0.2)

    w4 = tf.get_variable('d_w4', [3, 3, d3.get_shape()[1], 1024])
    d4 = tf.nn.conv2d(d3, w4, strides=[1, 1, 2, 2])
    d4 = tf.contrib.layers.batch_norm(d4, epsilon=1e-5, scope='d_b3')
    d4 = tf.nn.leaky_relu(d4, 0.2)

    flat = tf.reshape(d4, [-1, np.prod(d4.get_shape().as_list()[1:])])

    w5 = tf.get_variable('d_w5', [d4.get_shape()[-1], 2], initializer=tf.random_normal_initializer(stddev=0.02))
    d5 = tf.matmul(flat, w5)
    return d5


with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(None, 1))
    G = generator(z)

with tf.variable_scope('D') as scope:
    x = tf.placeholder(tf.float32, shape=(None, 1))
    D1 = discriminator(x)
    scope.reuse_variables()
    D2 = discriminator(G)


# TODO: may need to redo losses
loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
loss_g = tf.reduce_mean(-tf.log(D2))


# TODO: change optimizer. Adam?
def optimizer(loss, var_list):
    initial_learning_rate = 0.005
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(N)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer


vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

opt_d = optimizer(loss_d, d_params)
opt_g = optimizer(loss_g, g_params)

d_loss_record = np.zeros(num_steps)
g_loss_record = np.zeros(num_steps)

# TODO: load data from Impressionism folder and feed the nets
# with tf.Session() as session:
#     tf.global_variables_initializer().run()
#
#     for step in range(num_steps):
        # Update discriminator
        # xi = Real data sample
        # zi = Noise vector
        # d_loss_record[step], _ = session.run((loss_d, opt_d), {
        #     x: np.reshape(xi, (N, 1)),
        #     z: np.reshape(zi, (N, 1))
        # })

        # Update generator
    #     g_loss_record[step], _ = session.run((loss_g, opt_g), {
    #         z: np.reshape(zi, (N, 1))
    #     })
    #
    # print("Generator error:")
    # print(g_loss_record)
    # print("Output generator:")
    # print(session.run(G, {z: np.reshape(random.sample(10), (10, 1))}))