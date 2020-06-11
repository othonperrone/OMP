#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si 

# %%
start_date = '2014-01-01'
end_date = '2014-12-31'

# choose stock
R =si.get_data('AAPL',start_date=start_date,end_date=end_date)
R =pd.DataFrame(data=R.close, index=R.index)
R.rename(columns={'close':'AAPL'},inplace=True) 
R = R.pct_change()[1:]

# risk-free proxy
R_F = si.get_data('BIL',start_date=start_date,end_date=end_date)
R_F =pd.DataFrame(data=R_F.close, index=R_F.index)
R_F.rename(columns={'close':'R_F(BIL)'},inplace=True) 
R_F= R_F.pct_change()[1:]

# find it's beta against market
M = si.get_data('SPY',start_date=start_date,end_date=end_date)
M =pd.DataFrame(data=M.close, index=M.index)
M.rename(columns={'close':'M(SPY)'},inplace=True) 
M = M.pct_change()[1:]
#%%
R_P = R.subtract(pd.Series(data = R_F['R_F(BIL)'],index = R_F.index),axis=0)

#%%
AAPL_results = regression.linear_model.OLS(R_P, sm.add_constant(M)).fit()
AAPL_beta = AAPL_results.params[1]

#%%
plt.plot(M, label = 'M')
plt.plot(R,label = 'R')
plt.plot(R_F,label = 'R_F')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Daily Percent Return')

#%%
AAPL_results.summary()

# %%
R_Fs = pd.Series(data = R_F['R_F(BIL)'], index = R_F.index)
#%%
Ms = pd.Series(data = M['M(SPY)'], index = M.index)
predictions = R_Fs + AAPL_beta*(Ms - R_Fs) # CAPM equation

#%%
plt.plot(predictions)
plt.plot(R,color='Y')
plt.legend(['Prediction', 'Actual Return'])
plt.xlabel('Time')
plt.ylabel('Daily Percent Return');


# %%
