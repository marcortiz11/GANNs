import numpy as np
import scipy
from scipy import stats

def KL_divergence(d1,d2):
	#Step 1: Compute the probability density function
	left = min(d1+d2)
	right = max(d1+d2)
	density1 = np.histogram(d1,bins=20,range=(left,right),density=True)
	density2 = np.histogram(d2,bins=20,range=(left,right),density=True)
	#Step 2: Compute the KL divergence
	KLD = scipy.stats.entropy(density1[0],density2[0])
	return KLD
