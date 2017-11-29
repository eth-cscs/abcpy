#Backend
from abcpy.backends import BackendDummy
backend = BackendDummy()

import numpy as np
from continuousmodels import Normal, Uniform
from discretemodels import Binomial
from perturbationkernel import *
from summaryselections import Semiautomatic

# Simulate observed data
Y = Normal([2, 4])
Y_obs = Y.sample_from_distribution(1)[1].tolist()
print(Y_obs)
Y_obs_new = Y.sample_from_distribution(1)[1].tolist()
print(Y_obs_new)
#Y_obs = [Y_obs_1, Y_obs_2]


#Simple graphical model
theta = Normal([0, 1]) #list or float?
#w = Binomial([2, .5])
w = Uniform([[10], [20]])
sigma1 = Normal([w,20])
mu1 = Normal([theta, 1])
Y = Normal([mu1, sigma1])
#sigma2 = Normal([w,20])
#mu2 = Normal([theta,1])
#Y2 = Normal([mu2,sigma2])



# Note ??
#theta.sample_parameters()
#mu1.sample_parameters()
#w.sample_parameters()
#sigma1.sample_parameters()
#Y.sample_parameters()
#Y_obs_1 = Y.get_parameters().tolist()
#print(Y_obs)
#print(type(Y_obs))
#Y_obs = Y_obs.tolist()


#Statistics and Distance
from abcpy.statistics import Identity
from abcpy.distances import Euclidean
stat_calc = Identity(degree=3)
dist_calc = Euclidean(stat_calc)

print(stat_calc.statistics(Y_obs_new))
print(dist_calc.distance([Y_obs],[Y_obs_new]))

SA = Semiautomatic([Y], stat_calc, backend, n_samples = 2, seed = 1)

print('checked')









