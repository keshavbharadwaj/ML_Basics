import pandas as pd
df1=pd.DataFrame({'day':[1,2,3,4,5,6,7],'Life':[10,20,30,40,50,32,44],
                  'death':[2,5,4,3,6,7,8]},index=[100,200,300,400,500,600,700])

df2=pd.DataFrame({'day':[1,2,3,4,5,6,7],'Life':[10,20,30,40,50,32,44],
                  'death':[2,5,4,3,6,7,8]},index=[100,200,300,400,500,600,700])


df3=pd.DataFrame({'day':[1,2,3,4,5,6,7],'Problems':[10,20,30,40,50,32,44],
                  'Solutions':[2,5,4,3,6,7,8]},index=[100,200,300,400,500,600,700])

#print(df1)
print(pd.concat([df1,df2]))
df4=df1.append(df2)
print(df4)
