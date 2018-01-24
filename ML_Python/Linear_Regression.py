#Load Packages and set Directory
import matplotlib
matplotlib.__version__
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import seaborn
import statsmodels.formula.api as sm

#Load Data
import os
os.getcwd()
os.chdir("C:\\Users\\abhishek.b.jaiswal\\Desktop\\DataScience\\sem 2\\BD 3")
os.getcwd()
data=pd.read_csv("LR_1.csv", header=0)
print("Data shape:", data.shape)
print(data.head())
print(data.dtypes)

#Dummy creation
#Get all categorical variables and create dummies
obj=data.dtypes == np.object
print(obj)
dummydf=pd.DataFrame()

for i in data.columns[obj]:
    dummy=pd.get_dummies(data[i],drop_first=True)
    dummydf=pd.concat([dummydf,dummy],axis=1)
    

#Merge Dataset and dummy
data1=data
data1=pd.concat([data1,dummydf],axis=1)
print("head \n",data1.head())
obj1=data1.dtypes == np.object
data1=data1.drop(data1.columns[obj1],axis=1)
print("head after removal \n",data1.head())


#Declare the dependent varaiables and create your independent and dependant datasets
dep='House Price'
X=data1.drop(dep, axis=1)
Y=data1[dep]

#Scater plot
seaborn.pairplot(data1,kind='reg')

#Train and Test
import sklearn.cross_validation
#split into train and test
X_train,X_test,Y_train,Y_test=sklearn.cross_validation.train_test_split(X,Y,test_size=0.20,random_state=5)

#Run model
lm=sm.OLS(Y_train,X_train).fit()
lm.summary()
    

#Predict Train
pred_train=lm.predict(X_train)
err_train=pred_train -Y_train
#Predict Test
pred_test=lm.predict(X_test)
err_test=pred_test -Y_test

#Actuals vs predict plot
plt.scatter(Y_train,pred_train)
plt.xlabel('Y')
plt.ylabel('Pred')
plt.title('Main')

#Root Mean sq error
rmse=np.sqrt(np.mean((err_test))**2)
rmse

#Residual plot
plt.scatter(pred_train,err_train,c="b",s=40,alpha=0.5)
plt.scatter(pred_test,err_test,c="g",s=40)
plt.hlines(y=0,xmin=0,xmax=500)
plt.title('Residual plot - Train(blue),Test(green)')
plt.ylabel('Residuals')

