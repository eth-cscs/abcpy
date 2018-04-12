
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

def plot(samples, path = None, true_value = 5, title = 'ABC posterior'): 
	Bayes_estimate = np.mean(samples, axis = 0)
	theta = true_value
	xmin, xmax = max(samples[:,0]), min(samples[:,0])
	positions = np.linspace(xmin, xmax, samples.shape[0])
	gaussian_kernel = gaussian_kde(samples[:,0].reshape(samples.shape[0],))
	values = gaussian_kernel(positions)
	plt.figure()
	plt.plot(positions,gaussian_kernel(positions))
	plt.plot([theta, theta],[min(values), max(values)+.1*(max(values)-min(values))])
	plt.plot([Bayes_estimate, Bayes_estimate],[min(values), max(values)+.1*(max(values)-min(values))])
	plt.ylim([min(values), max(values)+.1*(max(values)-min(values))])
	plt.xlabel(r'$\theta$')
	plt.ylabel('density')
	#plt.xlim([0,1])
	plt.rc('axes', labelsize=15) 
	plt.legend(loc='best', frameon=False, numpoints=1)
	font = {'size'   : 15}
	plt.rc('font', **font)
	plt.title(title)
	if path is not None :
		plt.savefig(path)
	return plt



