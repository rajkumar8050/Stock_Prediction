import math
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#importing is over now read the data
dataset=pd.read_csv('E:\\stock_aal.csv')
X_value=dataset.iloc[:,0].values
Y_value=dataset.iloc[:,4].values
X=dataset.iloc[:,1:4].values
Y=dataset.iloc[:,4].values
# divid train and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
#create an object
lr=LinearRegression(normalize=True)
#training
lr.fit(X_train,Y_train)
y=lr.predict(X_test)  #that store the predicted value of X_test 
#so our predicted output is came so now let us check the probability of correctness by  R^2 algorithm
def r_square(Y_test,Y):
    ss_res=0
    ss_avg=0
    for i in range(len(Y_test)):
        ss_res+=pow(Y[i]-Y_test[i],2)
        ss_avg+=pow(Y[i]-np.mean(Y_test),2)
    r_sq=1-(ss_res/ss_avg)
    n=Y_test.shape[0]
    p=X.shape[1]
    y_res=(1-r_sq)*((n-1)/(n-p-1)) # n is the no of test case and p is no of attribute 
    res=1-y_res
    return res

print("Accuracy = %0.9f"%r_square(Y_test,y)) #this gives us the predicted output is how close to actual input
#split the date by year value
pq=[datetime.datetime.strptime(date,'%Y-%m-%d').date() for date in X_value[1007:1259]]
dt=np.arange(pq[0],pq[-1],330)
#plot the points
plt.figure(figsize=(12,10))
plt.scatter(pq,Y_test,color='red',label='actual value')
plt.scatter(pq,y,color='green',marker='^',label='predicted value')
plt.plot(pq,y,color='blue',label='predicted value')
plt.xticks(dt)
plt.xlabel("Year")
plt.ylabel("Closing Stock Price")
plt.title("Stock Prediction")
plt.legend()
plt.show()
#end of the code