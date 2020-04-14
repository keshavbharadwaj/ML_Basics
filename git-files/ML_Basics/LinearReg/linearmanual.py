##Note
##
##y=mx+b
##
##m=(Xb*Yb - XYb)/(xb)^2-(x^2)b
##
##b=yb-m*xb

##Note Rsquared THEORY
##Rsquared=1-SE(yhat)/SE(yb)
#R^2 is also  called coefficient of determination
#calculating R^2 to see how good the best fit line is
import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
def create_dataset(hm,variance,step=2,correlation=False):
    val=1
    ys=[]
    for i in range(hm):
        y=val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation=='pos':
            val+=step
        elif correlation=='neg':
            val-=step
    xs=[i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)
    
def fitterslope(xs,ys):
    m=(mean(xs)*mean(ys)-mean(xs*ys))/(mean(xs)**2-mean(xs**2))
    return m
def yintercept(xs,ys,m):
    b=mean(ys)-m*mean(xs)
    return b
def square_error_calc(ys_original,ys_line):
    return sum((ys_line-ys_original)**2)

def coefficient_of_determination(ys_original,ys_line):
    y_mean_line=[]
    for _ in ys_original:
        y_mean_line.append(mean(ys_original))
    SEyb=square_error_calc(ys_original,y_mean_line)
    SEyhat=square_error_calc(ys_original,ys_line)
    return 1 -(SEyhat/SEyb)


#xs=np.array([1,2,3,4,5,6],dtype=np.float64)
#ys=np.array([5,4,6,5,6,7],dtype=np.float64)
xs,ys=create_dataset(40,80,2,correlation="pos")
m=fitterslope(xs,ys)
b=yintercept(xs,ys,m)
#print(m,b)

regression_line=[(m*x+b) for x in xs]

predict_x=8
predict_y=m*predict_x+b

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g')
plt.plot(xs,regression_line)
plt.show()

r_squared=coefficient_of_determination(ys,regression_line)
print(r_squared)


