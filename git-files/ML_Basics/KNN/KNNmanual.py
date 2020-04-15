##Note
##euclidian distance
##sqrt((x-x1)**2+(y-y1)**2..................)
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as  pd
import random

style.use('fivethirtyeight')

def KNN(data,predict,k=3):
    if len(data)>=k:
        warnings.warn('K value is too small brah !!')
    distance=[]
    for group in data:
        for feat in data[group]:
            euclid_dist=np.linalg.norm(np.array(feat)-np.array(predict))
            #euclid_dist=sqrt(sum([(feat[i]-predict[i])**2 for i in range(len(feat))]))
            distance.append([euclid_dist,group])
    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result=Counter(votes).most_common(1)[0][0]
    confidence= Counter(votes).most_common(1)[0][1]/k
    return vote_result,confidence

##dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
##new_features=[5,7]
##result= KNN(dataset,new_features,k=3)
##print(result)
##
##[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]]for i in dataset]
##plt.scatter(new_features[0],new_features[1],s=100,color=result)
##plt.show()

df=pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data=df.astype(float).values.tolist() #we need to make sure everything is float some values are strings
#print(full_data[:10])
#print(100*'_')
random.shuffle(full_data)
#print(full_data[:10])

test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        vote,confidence=KNN(train_set,data,k=5)
        if group==vote:
            correct+=1
        else:
            print(confidence)
        total+=1
print('Accuracy : ',correct/total)



#each datapoint is independent so you can thread KNN heavily so thats why their KNN is Faster


















