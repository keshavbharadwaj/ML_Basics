#http://archive.ics.uci.edu/ml/datasets.php
from sklearn import preprocessing,neighbors,model_selection
import numpy as np
import pandas as pd
df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop('id',1,inplace=True)

X=np.array(df.drop(['class'],1))
Y=np.array(df['class'])

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)
accuracy=clf.score(X_test,Y_test)
print(accuracy)

example_measures=np.array([4,2,1,1,1,2,3,2,1])
#print(example_measures)
example_measures=example_measures.reshape(1,-1)
#example_measures=example_measures.reshape(len(example_measures),-1)
#print(example_measures)
prediction=clf.predict(example_measures)
print(prediction)
