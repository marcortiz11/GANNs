import tensorflow as tf
import numpy as np

"""
First example with GANNs. Training the system to emulate a Gaussian distribution
@Author: Marc Ortiz
"""

#Parameters
hidden_size = 500
num_steps = 1000
N = 20



#Create a Gaussian distribution for the discriminator to learn
class DataDistribution():
	def __init__(self,mu,sigma):
		self.mu = mu
		self.sigma = sigma
	
	def sample(self, N):
		samples = np.random.normal(self.mu,self.sigma,N)
		samples.sort()
		return samples


#This is the generator input noise distribution
class GeneratorDistribution():
	def __init__(self, range):
		self.range=range

	def sample(self,N):
		samples = np.linspace(-self.range,self.range,N) + np.random.random(N) * 0.01
		return samples.astype(np.float32)

#We define the generator
def generator(input, hidden_size):
	h0 = tf.nn.softplus(tf.layers.dense(input,hidden_size,name='g0'))
	h1 = tf.layers.dense(h0,1,name='g1')
	return h1


#Discriminator
def discriminator(input,hidden_size):
	h0 = tf.tanh(tf.layers.dense(input,hidden_size*2,name='d0'))
	h1 = tf.tanh(tf.layers.dense(h0,hidden_size*2,name='d1'))
	h2 = tf.tanh(tf.layers.dense(h1,hidden_size*2,name='d2'))
	h3 = tf.sigmoid(tf.layers.dense(h2,1,name='d3'))
	return h3


with tf.variable_scope('G'):
	z = tf.placeholder(tf.float32, shape=(None,1))
	G = generator(z,hidden_size)

with tf.variable_scope('D') as scope:
	x = tf.placeholder(tf.float32,shape=(None,1))
	D1 = discriminator(x,hidden_size)
	scope.reuse_variables()
	D2 = discriminator(G,hidden_size)

loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1-D2))
loss_g = tf.reduce_mean(-tf.log(D2))


def optimizer(loss,var_list):
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
		global_step = batch,
		var_list = var_list
	)
	return optimizer

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

opt_d = optimizer(loss_d,d_params)
opt_g = optimizer(loss_g,g_params)


data = DataDistribution(0,0.5)
random = GeneratorDistribution(10)


d_loss_record = np.zeros(num_steps)
g_loss_record = np.zeros(num_steps)


with tf.Session() as session:
	tf.global_variables_initializer().run()
	
	for step in range(num_steps):
		#Update discriminator
		xi = data.sample(N)
		zi = random.sample(N) 
		d_loss_record[step],_ = session.run((loss_d,opt_d), {
			x: np.reshape(xi,(N,1)),
			z: np.reshape(zi,(N,1))
		})
		

		#Update generator
		g_loss_record[step], _ = session.run((loss_g,opt_g),{
			z: np.reshape(zi,(N,1))
		})
	
	print("Generator error:")
	print(g_loss_record)
	print("Output generator:")
	print(session.run(G,{z:np.reshape(random.sample(10),(10,1))}))
