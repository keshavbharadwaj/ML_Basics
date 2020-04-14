import quandl
import pandas as pd
import numpy as np
import sys
import datetime
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing,svm,model_selection
from sklearn.linear_model import LinearRegression
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_column',None)
quandl.ApiConfig.api_key=ID
df=quandl.get('WIKI/GOOGL')
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100
df['PCT_Change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Close']*100
df=df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)


x = np.array(df.drop(['label'],1))
x=preprocessing.scale(x)
X=x[:-forecast_out]
X_latest=x[-forecast_out:]
df.dropna(inplace=True)
Y=np.array(df['label'])


X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.2)

clf=LinearRegression(n_jobs=-1)
#clf=svm.SVR()
clf.fit(X_train,Y_train)
accuracy=clf.score(X_test,Y_test)
#print(accuracy)
forecast_set=clf.predict(X_latest)
print(forecast_set,accuracy,forecast_out)
