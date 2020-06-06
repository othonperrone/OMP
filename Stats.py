#%% Import libraries
import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime
from yahoo_fin import stock_info as si 

#%%
def linreg(X,Y):
    # Running the linear regression
    X = sm.add_constant(X)
    model = regression.linear_model.OLS(Y, X).fit()
    a = model.params[0]
    b = model.params[1]
    X = X[:, 1]

    # Return summary of the regression and plot results
    X2 = np.linspace(X.min(), X.max(), 100)
    Y_hat = X2 * b + a
    plt.scatter(X, Y, alpha=0.3) # Plot the raw data
    plt.plot(X2, Y_hat, 'r', alpha=0.9) # Add the regression line, colored in red
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    return model.summary()

#%%
start = '2019-01-01'
end = '2020-01-01'
asset = pd.DataFrame(si.get_data('cogn3.sa',start_date=start,end_date=end))
benchmark = pd.DataFrame(si.get_data('BOVA11.SA', start_date=start, end_date=end))
asset = pd.DataFrame(data=asset.close,index=asset.index)
benchmark = pd.DataFrame(data=benchmark.close,index=benchmark.index)
r_a = asset.pct_change()[1:]
r_b = benchmark.pct_change()[1:]
#1234

linreg(r_b.values, r_a.values)

# %%
