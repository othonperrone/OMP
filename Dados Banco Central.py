#%%CODIGO PARA PEGAR DADOS HISTORICOS DO BANCO CENTRAL COMO O IPCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%

#Link para encontrat codigos
#https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries
codigo_bcb = '433' #IPCA
url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.' + codigo_bcb + '/dados?formato=json'

#Link para entender
#https://dadosabertos.bcb.gov.br/dataset/20542-saldo-da-carteira-de-credito-com-recursos-livres---total/resource/6e2b0c97-afab-4790-b8aa-b9542923cf88

df = pd.read_json(url)
df['data'] = pd.to_datetime(df['data'],dayfirst=True)
df.set_index('data',inplace=True)

IPCA = df.loc[df.index>='2018-01-01']
IPCA_adj = IPCA/100 + 1
IPCAacumulado = IPCA_adj.cumprod()
IPCAYTD = IPCA_adj.groupby(IPCA_adj.index.year).cumprod()

IPCA2 = IPCAacumulado
IPCA2['YTD'] = IPCAYTD.valor
IPCA2['Month'] = IPCA.valor/100
#print(df)
#plt.plot(df)


# %%
