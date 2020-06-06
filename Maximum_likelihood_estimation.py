#%% Import Libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats

# %%
TRUE_MEAN = 40
TRUE_STD = 10
X = np.random.normal(TRUE_MEAN, TRUE_STD, 1000)

# %%
def normal_mu_MLE(X):
    # Get the number of observations
    T = len(X)
    # Sum the observations
    s = sum(X)
    return 1.0/T * s

def normal_sigma_MLE(X):
    T = len(X)
    # Get the mu MLE
    mu = normal_mu_MLE(X)
    # Sum the square of the differences
    s = sum( np.power((X - mu), 2) )
    # Compute sigma^2
    sigma_squared = 1.0/T * s
    return math.sqrt(sigma_squared)

# %%
print("Mean Estimation")
print(normal_mu_MLE(X))
print(np.mean(X))
print("Standard Deviation Estimation")
print(normal_sigma_MLE(X))
print(np.std(X))

# %%
mu, std = scipy.stats.norm.fit(X)
print("mu estimate: " + str(mu))
print("std estimate: " + str(std))

# %%
pdf = scipy.stats.norm.pdf
# We would like to plot our data along an x-axis ranging from 0-80 with 80 intervals
# (increments of 1)
x = np.linspace(0, 80, 80)
plt.hist(X, bins=x, normed='true')
plt.plot(pdf(x, loc=mu, scale=std))
plt.xlabel('Value')
plt.ylabel('Observed Frequency')
plt.legend(['Fitted Distribution PDF', 'Observed Data', ]);

# %%
