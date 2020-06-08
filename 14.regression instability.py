#%%Importing libraries
import numpy as np
import pandas as pd
from statsmodels import regression, stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
from yahoo_fin import stock_info as si 

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

#%%
def linreg(X,Y):
    # Running the linear regression
    x = sm.add_constant(X) # Add a row of 1's so that our model has a constant term
    model = regression.linear_model.OLS(Y, x).fit()
    return model.params[0], model.params[1] # Return the coefficients of the linear model

#%%
# Draw observations from normal distribution
np.random.seed(107) # Fix seed for random number generation
rand = np.random.randn(20)
#%%
# Conduct linear regression on the ordered list of observations
xs = np.arange(20)
a, b = linreg(xs, rand)
print('Slope:', b, 'Intercept:', a)

#%%
# Plot the raw data and the regression line
plt.scatter(xs, rand, alpha=0.7)
Y_hat = xs * b + a
plt.plot(xs, Y_hat, 'r', alpha=0.9);

# %%
import seaborn
seaborn.regplot(xs, rand)

# %%
# Draw more observations
rand2 = np.random.randn(100)

# Conduct linear regression on the ordered list of observations
xs2 = np.arange(100)
a2, b2 = linreg(xs2, rand2)
print('Slope:', b2, 'Intercept:', a2)

# Plot the raw data and the regression line
plt.scatter(xs2, rand2, alpha=0.7)
Y_hat2 = xs2 * b2 + a2
plt.plot(xs2, Y_hat2, 'r', alpha=0.9);


# %%

start = '2003-01-01'
end = '2009-02-01'
pricing = si.get_data('SPY',start_date=start,end_date=end)
pricing = pd.DataFrame(data=pricing.close,index=pricing.index)
# Manually set the point where we think a structural break occurs
breakpoint = 1200
xs = np.arange(len(pricing))
xs2 = np.arange(breakpoint)
xs3 = np.arange(len(pricing) - breakpoint)

# Perform linear regressions on the full data set, the data up to the breakpoint, and the data after
a, b = linreg(xs, pricing)
a2, b2 = linreg(xs2, pricing[:breakpoint])
a3, b3 = linreg(xs3, pricing[breakpoint:])

Y_hat = pd.Series(xs * b + a, index=pricing.index)
Y_hat2 = pd.Series(xs2 * b2 + a2, index=pricing.index[:breakpoint])
Y_hat3 = pd.Series(xs3 * b3 + a3, index=pricing.index[breakpoint:])

# Plot the raw data
pricing.plot()
Y_hat.plot(color='y')
Y_hat2.plot(color='r')
Y_hat3.plot(color='r')
plt.title('SPY Price')
plt.ylabel('Price');

#%% Charts with Plotly
PRICE = go.Scatter(x = pricing.index,
            y = pricing.close,
            mode = 'lines')

Y_hatpy = go.Scatter(x = pricing.index,
            y = xs * b + a,
            mode = 'lines')

Y_hatpy2 = go.Scatter(x = pricing.index[:breakpoint],
            y = xs2 * b2 + a2,
            mode = 'lines')

Y_hatpy3 = go.Scatter(x = pricing.index[breakpoint:],
            y = xs3 * b3 + a3,
            mode = 'lines')

data = [PRICE, Y_hatpy,Y_hatpy2,Y_hatpy3]
py.iplot(data)

# %%NAO ENTENDIII
stats.diagnostic.breaks_cusumolsresid(
    regression.linear_model.OLS(pricing, sm.add_constant(xs)).fit().resid)[1]


# %%
# Get pricing data for two benchmarks (stock indices) and a stock
start = '2013-01-01'
end = '2015-01-01'
b1 = si.get_data('SPY',start_date=start,end_date=end)
b1 = pd.DataFrame(data=b1.close,index=b1.index)
b2 = si.get_data('MDY',start_date=start,end_date=end)
b2 = pd.DataFrame(data=b2.close,index=b2.index)
asset = si.get_data('V',start_date=start,end_date=end)
asset = pd.DataFrame(data=asset.close,index=asset.index)

#%%%


print(sm.add_constant(np.column_stack((b1, b2))))
#%%1

mlr = regression.linear_model.OLS(asset, sm.add_constant(np.column_stack((b1, b2)))).fit()
prediction = mlr.params[0] + mlr.params[1]*b1 + mlr.params[2]*b2
print('Constant:', mlr.params[0], 'MLR beta to S&P 500:', mlr.params[1], ' MLR beta to MDY', mlr.params[2])

# Plot the asset pricing data and the regression model prediction, just for fun
plt.plot(asset)
plt.plot(prediction)
#asset.plot()
#prediction.plot(color='y')
plt.ylabel('Price')
plt.legend(['Asset', 'Linear Regression Prediction']);

# %s%

sp.stats.pearsonr(b1.close,b2.close)[0]



# %%
