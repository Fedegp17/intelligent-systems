import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
#import the data frame we are gonna be using and inspect first 5 rows 
df=pd.read_csv("Summary of Weather.csv")
print(df.head())
print(df.describe())

#Search for the correct variable correalation
corr= df.corr()
print(corr)
pd.plotting.scatter_matrix(df, alpha=0.2)
plt.show()
#convert the data frame into a numpy matrix
dfn=df.to_numpy()

#split the data 
x=dfn[:,[5,6]] #Indexing values in the dataframe
y=dfn[:,[4]] #target value, in this case the highest value of the day


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

#create the linear regression object
regr = linear_model.LinearRegression()

#train the model using the training sets
regr.fit(x_train, y_train)

#make predictions using the testing set
y_pred= regr.predict(x_test)

#get the coef
print('Coefficients: \n', regr.coef_)

#MSE
print('Mean squared error: %.2f '%mean_squared_error(y_test, y_pred))

#r2 coefficient determination
print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))

plt.scatter(x_train, y_train,color='g') 

plt.plot(x_test , y_pred,color='k')

plt.show()