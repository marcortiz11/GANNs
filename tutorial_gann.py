import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import utils

sns.set(color_codes=True)


"""
First example with GANNs. Training the system to approximate a Gaussian distribution
@Author: Marc Ortiz
"""

#Parameters
hidden_size = 100
num_steps = 10000
N = 50 #batch size
display_step = num_steps/100 #Displaying the dist. of the generator each display_step iterations
K = 1 #Higher value makes the discriminator smarter than the generator


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
		return samples
    

#Generator
def generator(input, hidden_size):
	h0 = tf.nn.tanh(tf.layers.dense(input,hidden_size,name='g0'))
	h1 = tf.nn.tanh(tf.layers.dense(h0,hidden_size,name='g1'))
	#h2 = tf.nn.softplus(tf.layers.dense(h1,hidden_size,name='g2'))
	h3 = tf.layers.dense(h1,1,name='g3')
	return h3


#Discriminator
def discriminator(input,hidden_size):
	h0 = tf.tanh(tf.layers.dense(input,hidden_size*2,name='d0'))
	h1 = tf.tanh(tf.layers.dense(h0,hidden_size*2,name='d1'))
	h2 = tf.tanh(tf.layers.dense(h1,hidden_size*2,name='d2'))
	#h3 = tf.tanh(tf.layers.dense(h2,hidden_size*2,name='d3'))
	#h4 = tf.tanh(tf.layers.dense(h3,hidden_size*2,name='d4'))
	h5 = tf.sigmoid(tf.layers.dense(h2,1,name='d5'))
	return h5


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
	initial_learning_rate = 0.007
	decay = 0.95
	num_decay_steps = 200
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



####### TRAINING #########
data = DataDistribution(2,2)
random = GeneratorDistribution(5)

d_loss_record = np.zeros(num_steps)
g_loss_record = np.zeros(num_steps)
KL_div_record = np.zeros(num_steps)

data_plot = data.sample(1000)
random_plot = random.sample(1000)


plt.ion()
#plt.plot(data_plot)
plt.show()

with tf.Session() as session:
	tf.global_variables_initializer().run()
	
	for step in range(num_steps):
		xi = data.sample(N)
		zi = random.sample(N) 
		for k in range(K):
			#Train K times the discriminator
			#to make it ahead of the generator
			d_loss_record[step],_ = session.run((loss_d,opt_d), {
				x: np.reshape(xi,(N,1)),
				z: np.reshape(zi,(N,1))
			})
		
		#Update generator
		#zi = random.sample(N) 
		g_loss_record[step], _,dist1 = session.run((loss_g,opt_g,G),{
			z: np.reshape(zi,(N,1))
		})

		KL_div_record[step] = utils.KL_divergence(xi,dist1.flatten())

		if (step%display_step == 0):    
			dist = session.run(G,{z:np.reshape(random_plot,(1000,1))})
			plt.cla()
			sns.distplot(data_plot,hist=False, rug=False)
			sns.distplot(dist,hist=False, rug=False, color='r')
			plt.draw()
			plt.pause(0.1)
		
		

	#We need to print the generator and discriminator errors
	print("Generator error:")
	plt.plot(d_loss_record)
	plt.plot(g_loss_record)
	#plt.plot(KL_div_record)
	#print(KL_div_record)
	plt.show()

	print("Generator learned distribution:")
	dist_final_approx = session.run(G,{z:np.reshape(random_plot,(1000,1))})
	dist_original = data_plot
	print("mean:", np.mean(dist_final_approx))
	print("std:", np.std(dist_final_approx))

	print("Similarity between the two distributions:")
	print(KL_div_record[-1])

