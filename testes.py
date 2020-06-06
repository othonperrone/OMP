#%% Importando libraries
import pandas as pd 
from yahoo_fin import stock_info as si 
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt 
#s

from sklearn.linear_model import LinearRegression as ln

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

import statsmodels.api as sm
#%% definição da data inicial
startdate = dt.date(2019,1,1)
#%%
cogn3 = si.get_data('cogn3.sa',start_date=startdate)
yduq3 = si.get_data('yduq3.sa',start_date=startdate)
BVSP = si.get_data('^BVSP',start_date=startdate)
#%%
cogna = pd.DataFrame(data=cogn3.close,index=cogn3.index)
yduqs = pd.DataFrame(data=yduq3.close,index=yduq3.index)
IBOV = pd.DataFrame(data=BVSP.close,index=BVSP.index)
change_cogn = cogna.pct_change()
change_yduqs = yduqs.pct_change()

#%%
plt.subplot(2,1,1)
plt.hist(change_cogn.close,bins=20,cumulative=True)
plt.title("Cogn")
plt.subplot(2,1,2)
plt.hist(change_yduqs.close,bins=20,cumulative=True)

#%%
cogna.fillna(method='bfill',inplace=True)
yduqs.fillna(method='bfill',inplace=True)
IBOV.fillna(method='bfill',inplace=True)
cogna_fit = np.array(cogna.close).reshape(-1,1)
yduqs_fit= np.array(yduqs.close).reshape(-1,1)
plt.scatter(cogna_fit, yduqs_fit)
#%%
model = ln().fit(cogna_fit,yduqs_fit)
yduqs_pred = model.predict(cogna_fit)
plt.scatter(cogna_fit, yduqs_fit)
plt.plot(cogna_fit,yduqs_pred)
# %%
print(np.corrcoef(cogna.close,IBOV.close))
print(np.corrcoef(yduqs.close,IBOV.close))
# %%
COGNA = go.Scatter(x = cogna.index ,
                   y = cogna.close ,
                   mode = 'lines')

YDUQS = go.Scatter(x = yduqs.index ,
                   y = yduqs.close ,
                   mode = 'lines')

data = [COGNA,YDUQS]
py.iplot(data)

# %%
