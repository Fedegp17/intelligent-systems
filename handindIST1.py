import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from LinearRegression import LinearRegression

#import the data set and evaluate it
df=pd.read_csv("Tesla_stock_Price.csv")
print(df.head())


#find the correlation among the features
corr= df.corr()
#print(corr)
#convert the data frame into a numpy matrix
dfn=df.to_numpy()
print(dfn.shape)

#split the data 
x=dfn[:,[3]] #Indexing all numerical values in the dataframe
y=dfn[:,2] #target value, in this case the chighest value of the day

#split the data in training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

fig = plt.figure(figsize=(8,6))

plt.scatter(x[:,0],y,color="g", marker = "o", s=30)
#plt.show()

reg= LinearRegression(lr=0.01)
reg.fit(x_train, y_train)
y_pred= reg.predict(x_test)
mse = reg.mse(y_test,y_pred)
r2= reg.r_squared(y_test, y_pred)

y_pred_line = reg.predict(x)
cmap= plt.get_cmap('viridis')
m1= plt.scatter(x_train, y_train, color= cmap(0.9), s=10)
m2= plt.scatter(x_test, y_test, color= cmap(0.5), s=10)
plt.plot(x, y_pred_line, color = "b", linewidth=2, label="Prediction")
#plt.show()
print(mse)
print(r2)
