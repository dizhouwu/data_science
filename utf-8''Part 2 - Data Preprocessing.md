
# Module 5: Regime Prediction with Machine Learning - Part 2

In this part we will prepare the dataset for our recession forecasting problem. We will clean the data and perform feature selection to reduce the number of variables in the data.

## Table of Contents:
&nbsp;&nbsp;1. [Set Up Environment and Read Data](#1)

&nbsp;&nbsp;2. [Data Cleaning](#2)


## 1. Set Up Environment and Read Data <a id="1"></a>


```python
#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from statsmodels.tsa.stattools import adfuller #to check unit root in time series 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

import seaborn as sns #for correlation heatmap

import warnings
warnings.filterwarnings('ignore')
```


```python
bigmacro=pd.read_csv("Macroeconomic_Variables.csv")
bigmacro=bigmacro.rename(columns={'sasdate':'Date'})
Recession_periods=pd.read_csv('Recession_Periods.csv')
bigmacro.insert(loc=1,column="Regime", value=Recession_periods['Regime'].values)
bigmacro.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Regime</th>
      <th>RPI</th>
      <th>W875RX1</th>
      <th>DPCERA3M086SBEA</th>
      <th>CMRMTSPLx</th>
      <th>RETAILx</th>
      <th>INDPRO</th>
      <th>IPFPNSS</th>
      <th>IPFINAL</th>
      <th>...</th>
      <th>DSERRG3M086SBEA</th>
      <th>CES0600000008</th>
      <th>CES2000000008</th>
      <th>CES3000000008</th>
      <th>UMCSENTx</th>
      <th>MZMSL</th>
      <th>DTCOLNVHFNM</th>
      <th>DTCTHFNM</th>
      <th>INVEST</th>
      <th>VXOCLSx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/59</td>
      <td>Normal</td>
      <td>2437.296</td>
      <td>2288.8</td>
      <td>17.302</td>
      <td>292258.8329</td>
      <td>18235.77392</td>
      <td>22.6248</td>
      <td>23.4555</td>
      <td>22.1893</td>
      <td>...</td>
      <td>11.358</td>
      <td>2.13</td>
      <td>2.45</td>
      <td>2.04</td>
      <td>NaN</td>
      <td>274.9</td>
      <td>6476.0</td>
      <td>12298.0</td>
      <td>84.2043</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/1/59</td>
      <td>Normal</td>
      <td>2446.902</td>
      <td>2297.0</td>
      <td>17.482</td>
      <td>294429.5453</td>
      <td>18369.56308</td>
      <td>23.0679</td>
      <td>23.7720</td>
      <td>22.3816</td>
      <td>...</td>
      <td>11.375</td>
      <td>2.14</td>
      <td>2.46</td>
      <td>2.05</td>
      <td>NaN</td>
      <td>276.0</td>
      <td>6476.0</td>
      <td>12298.0</td>
      <td>83.5280</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3/1/59</td>
      <td>Normal</td>
      <td>2462.689</td>
      <td>2314.0</td>
      <td>17.647</td>
      <td>293425.3813</td>
      <td>18523.05762</td>
      <td>23.4002</td>
      <td>23.9159</td>
      <td>22.4914</td>
      <td>...</td>
      <td>11.395</td>
      <td>2.15</td>
      <td>2.45</td>
      <td>2.07</td>
      <td>NaN</td>
      <td>277.4</td>
      <td>6508.0</td>
      <td>12349.0</td>
      <td>81.6405</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4/1/59</td>
      <td>Normal</td>
      <td>2478.744</td>
      <td>2330.3</td>
      <td>17.584</td>
      <td>299331.6505</td>
      <td>18534.46600</td>
      <td>23.8987</td>
      <td>24.2613</td>
      <td>22.8210</td>
      <td>...</td>
      <td>11.436</td>
      <td>2.16</td>
      <td>2.47</td>
      <td>2.08</td>
      <td>NaN</td>
      <td>278.1</td>
      <td>6620.0</td>
      <td>12484.0</td>
      <td>81.8099</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5/1/59</td>
      <td>Normal</td>
      <td>2493.228</td>
      <td>2345.8</td>
      <td>17.796</td>
      <td>301372.9597</td>
      <td>18679.66354</td>
      <td>24.2587</td>
      <td>24.4628</td>
      <td>23.0407</td>
      <td>...</td>
      <td>11.454</td>
      <td>2.17</td>
      <td>2.48</td>
      <td>2.08</td>
      <td>95.3</td>
      <td>280.1</td>
      <td>6753.0</td>
      <td>12646.0</td>
      <td>80.7315</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 130 columns</p>
</div>




```python
Recession_periods['Regime'].value_counts()
```




    Normal       628
    Recession     93
    Name: Regime, dtype: int64



## 2. Data Cleaning <a id="2"></a>

We will follow the steps below to clean data and make it ready for feature selection process.

1. Remove the variables with missing observations
2. Add lags of the variables as additional features
3. Test stationarity of time series
4. Standardize the dataset


```python
Recession_periods.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Date</th>
      <th>Regime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1/1/59</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2/1/59</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3/1/59</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4/1/59</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5/1/59</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove columns with missing observations
missing_colnames=[]
for i in bigmacro.drop(['Date','Regime'],axis=1):
    observations=len(bigmacro)-bigmacro[i].count()
    if (observations>10):
        print(i+':'+str(observations))
        missing_colnames.append(i)
 
bigmacro=bigmacro.drop(labels=missing_colnames, axis=1)

#rows with missing values
bigmacro=bigmacro.dropna(axis=0)

bigmacro.shape
```

    PERMIT:13
    PERMITNE:13
    PERMITMW:13
    PERMITS:13
    PERMITW:13
    ACOGNO:398
    ANDENOx:110
    TWEXMMTH:168
    UMCSENTx:155
    VXOCLSx:42





    (718, 120)




```python
# Add lags
for col in bigmacro.drop(['Date', 'Regime'], axis=1):
    for n in [3,6,9,12,18]:
        bigmacro['{} {}M lag'.format(col, n)] = bigmacro[col].shift(n).ffill().values

# 1 month ahead prediction
bigmacro["Regime"]=bigmacro["Regime"].shift(-1)

bigmacro=bigmacro.dropna(axis=0)
```


```python
bigmacro.shape
```




    (699, 710)



Augmented Dickey-Fuller Test can be used to test for stationarity in macroeconomic time series variables. We will use `adfuller` function from `statsmodels` module in Python. More information about the function can be found __[here](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)__.


```python
#check stationarity
from statsmodels.tsa.stattools import adfuller #to check unit root in time series 
threshold=0.01 #significance level
for column in bigmacro.drop(['Date','Regime'], axis=1):
    result=adfuller(bigmacro[column])
    if result[1]>threshold:
        bigmacro[column]=bigmacro[column].diff()
bigmacro=bigmacro.dropna(axis=0)
```


```python
threshold=0.01 #significance level
for column in bigmacro.drop(['Date','Regime'], axis=1):
    result=adfuller(bigmacro[column])
    if result[1]>threshold:
        bigmacro[column]=bigmacro[column].diff()
bigmacro=bigmacro.dropna(axis=0)
```


```python
threshold=0.01 #significance level
for column in bigmacro.drop(['Date','Regime'], axis=1):
    result=adfuller(bigmacro[column])
    if result[1]>threshold:
        print(column)
bigmacro=bigmacro.dropna(axis=0)      
```


```python
# Standardize
from sklearn.preprocessing import StandardScaler
features=bigmacro.drop(['Date','Regime'],axis=1)
col_names=features.columns

scaler=StandardScaler()
scaler.fit(features)
standardized_features=scaler.transform(features)
standardized_features.shape
df=pd.DataFrame(data=standardized_features,columns=col_names)
df.insert(loc=0,column="Date", value=bigmacro['Date'].values)
df.insert(loc=1,column='Regime', value=bigmacro['Regime'].values)
df.head()
df.shape
```


```python
df.to_csv("Dataset_Cleaned.csv", index=False)
```
