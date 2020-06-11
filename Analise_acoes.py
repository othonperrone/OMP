#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si 

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

import plotly.express as px
#%% setor
TBL_setor = pd.read_excel('Setorial.xlsx',sheet_name='Plan3')


#%%GSelecionando setor
Seg = 'Análises e Diagnósticos'
#Sec_A = TBL_setor[(TBL_setor.SEGMENTO==Seg)]
Sec_A = TBL_setor
#%%
start_date = '2020-02-26'
end_date = '2020-12-31'
# choose stock
Sec = Sec_A.iloc[:]['CÓDIGO']


#%%
Stocks = pd.DataFrame()
for i in range(len(Sec)):
    try:
        Stock =si.get_data((Sec.iloc[i]+'3.SA'),start_date=start_date,end_date=end_date)
        Stocks.insert(0,column=Sec.iloc[i]+'3.SA',value = Stock['close'])
        try:
            Stock =si.get_data((Sec.iloc[i]+'4.SA'),start_date=start_date,end_date=end_date)
            Stocks.insert(0,column=Sec.iloc[i]+'4.SA',value = Stock['close'])
        except:
            print(Sec.iloc[i]+'4.SA' + " not found")
    except:
        print(Sec.iloc[i]+'3.SA' + " not found")
# %%
# %%
Returns = Stocks.pct_change()
Cumulative_Returns = (Returns+1).cumprod()-1
#Cumulative_Returns.plot()
#plt.legend()

# %%
RET = px.line(Cumulative_Returns)
RET.show()

# %%
