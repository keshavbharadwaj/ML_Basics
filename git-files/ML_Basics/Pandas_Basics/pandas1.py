import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
my_site={'Day':[1,2,3,4,5,6],
         'Vistors':[50,40,59,54,65,56],
         'Bounce_Rate':[65,72,62,64,54,66]}
df=pd.DataFrame(my_site)
df.set_index('Day',inplace=True)
print(df)
print(df.Vistors.tolist())
df2=pd.read_csv('/home/keshav/Desktop/xc.csv')
print(df2.head())
