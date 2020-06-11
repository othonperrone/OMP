#%%

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
#%%
# Example of ranking data
l = [10, 9, 5, 7, 5]
print('Raw data: ', l)
print('Ranking: ', list(stats.rankdata(l, method='average')))

# %%

## Let's see an example of this
n = 100
#%%
def compare_correlation_and_spearman_rank(n, noise):
    X = np.random.poisson(size=n)
    Y = np.exp(X) + noise * np.random.normal(size=n)

    Xrank = stats.rankdata(X, method='average')
    # n-2 is the second to last element
    Yrank = stats.rankdata(Y, method='average')

    diffs = Xrank - Yrank # order doesn't matter since we'll be squaring these values
    r_s = 1 - 6*sum(diffs*diffs)/(n*(n**2 - 1))
    c_c = np.corrcoef(X, Y)[0,1]
    
    return r_s, c_c

#%%
experiments = 1000
spearman_dist = np.ndarray(experiments)
correlation_dist = np.ndarray(experiments)
for i in range(experiments):
    r_s, c_c = compare_correlation_and_spearman_rank(n, 1.0)
    spearman_dist[i] = r_s
    correlation_dist[i] = c_c
    
print('Spearman Rank Coefficient: ' + str(np.mean(spearman_dist)))
# Compare to the regular correlation coefficient
print('Correlation coefficient: ' + str(np.mean(correlation_dist)))

# %%Não TErminada

