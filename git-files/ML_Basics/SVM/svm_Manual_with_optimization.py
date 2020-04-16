import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class svm:
    def __init__(self,visual=True):
        self.visual=visual
        self.colors={1:'r',-1:'b'}
        if self.visual:
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(1,1,1)

    def fit(self,data):
        self.data=data
        #(||w||:[w,b])
        opt_dict={}

        transform=[[1,1],
                   [-1,1],
                   [-1,-1],
                   [1,-1]]

        all_data=[]
        for yi in self.data:
            for features in self.data[yi]:
                for f in features:
                    all_data.append(f)
                    
        self.max_feature_value=max(all_data)
        self.min_feature_value=min(all_data)
        step_sizes=[self.max_feature_value*0.1,self.max_feature_value*0.01,
                    #self.max_feature_value*0.025
                    ]

        #expensive to calculate B

        b_range_multiple=5
        b_multiple=5
        latest_optimum = self.max_feature_value*10
        #we are going to make W as [x,x] ie make all dimensions of W equal  in value
        #this is going to reduce the processing required,and gives us a fairly ok result
        #For the best we need to change each individual dimension of W

        for step in step_sizes:
            w=np.array([latest_optimum,latest_optimum])
            optimized=False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,step*b_multiple):
                    for transformation in transform:
                        w_t=w*transformation
                        found_option=True
                        #weakest link in SVM
                        #SMO tries to fix it
                        #constraint function yi(xi.w+b)>=1
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_option=False
                                    break
                            if(found_option==False):
                                break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)]=[w_t,b]
                if w[0]<0:
                    optimized = True
                    print('optimized a step')
                else:
                    w=w-[step,step]
            norms=sorted([n for n in opt_dict])
            opt_choice=opt_dict[norms[0]]
            self.w=opt_choice[0]
            self.b=opt_choice[1]
            latest_optium=opt_choice[0][0]+step*2                      
                                
    
    def predict(self,features):
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visual:
            self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #The hyper plane =x.w+b
        #v=x.w+b
        #psv=1
        #nsv=-1
        #dec=0
        def hyperplane(x,w,b,v):
             return(-w[0]*x-b+v)/w[1] #ok figured this thing out refer to the images for what this is returning

        datarange=(self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min=datarange[0]
        hyp_x_max=datarange[1]

        #(w.x+b)=1 psv hyperplane
        psv1 = hyperplane(hyp_x_min,self.w,self.b,1)
        #print(psv1)
        psv2 = hyperplane(hyp_x_max,self.w,self.b,1)
        #print(psv2)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')

        #(w.x+b)=-1 nsv hyperplane
        nsv1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')

        #(w.x+b)=0 decision hyperplane
        db1 = hyperplane(hyp_x_min,self.w,self.b,0)
        db2 = hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'k--')

        plt.show()

data_dict={-1:np.array([[1,7],
                        [2,8],
                        [3,8]]),
           1:np.array([[5,1],
                        [6,-1],
                        [7,3]])}




s=svm()
s.fit(data=data_dict)
predictions=[[1,3],
            [4,6],
            [5,9],
            [5,8],
            [7,6],
            [1,5],
            [3,4]]

for p in predictions:
    s.predict(p)
s.visualize()
