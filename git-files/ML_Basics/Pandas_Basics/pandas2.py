import quandl
import pandas as pd
quandl.ApiConfig.api_key=ID
df=quandl.get('FMAC/HPI_AK')
#print(df)
states=pd.read_html('https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States')
print(states)
