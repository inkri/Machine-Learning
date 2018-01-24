#Load Packages and set Directory
import numpy as py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm

#User defined function
def measures(Technique,Actual,Predicted,cutoff=0.5):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    
    confusionmat=metrics.confusion_matrix(Actual,Predicted)
    TP=confusionmat[0,0]
    TN=confusionmat[1,1]
    FP=confusionmat[0,1]
    FN=confusionmat[1,0]
    P=TP+FN
    N=TN+FP
    
    #Measures
    Acc=np.round((TP+TN)/(P+N),4)
    KS=np.round(abs((TP/P)-(FP/N)),4)
    Precision=np.round((TP/(TP+FP)),4)
    Recall=np.round((TP/(TP+FN)),4)
    
    Results=pd.DataFrame(columns=['Technique','Cutoff','Accuracy','KS','Precision','Recall'])
    Results=Results.append(pd.DataFrame([Technique,cutoff,Acc,KS,Precision,Recall],
                                       index=['Technique','Cutoff','Accuracy','KS','Precision','Recall']).T)
    return(Results)
    

#User defined function
def bestcutoff(actual,pred,Technique,cuts=4):
    import numpy as np
    seq=np.round(np.append(np.arange(0,100,cuts)/100,1),2)
    cutdata=pd.DataFrame(columns=['Technique','Cutoff','Accuracy','KS','Precision','Recall'])
    final=pd.DataFrame(columns=['Technique','Accuracy','KS','Precision','Recall'])
    for i in seq:
        predicted=pred.loc[:,0].map(lambda x: 1 if x > i else 0)
        templ=measures(Technique,actual,predicted,i)
        cutdata=cutdata.append(templ).fillna(0)
        
    cutdata=cutdata.set_index(np.arange(len(seq)))
    print(cutdata)
    temp=[]
    for j in range(2,6):
        print("temp",temp)
        temp=np.append(temp,cutdata.iloc[cutdata.iloc[:,j].idxmax(skipna=True),1])
        
    listval=list(np.append(Technique,temp))
    final.loc[len(final)]=listval
    
    print(Technique,"Best Cutoff Completed \n")
    print("################################ \n")
    return(final)    



#Load Data
import os
os.getcwd()
os.chdir("C:\\Users\\abhishek.b.jaiswal\\Desktop\\DataScience\\sem 2\\BD 3")
os.getcwd()
data=pd.read_csv("binary.csv", header=0)
print("Data shape:", data.shape)
print(data.head())
print(data.dtypes)


#Convert categorical variables with numeric values to str type
data['rank']=data['rank'].astype(str)

#Get all categorical variables and create dummies
obj=data.dtypes == py.object
dep='admit'
obj[dep]=False
dummydf=pd.DataFrame()

for i in data.columns[obj]:
    dummy=pd.get_dummies(data[i],drop_first=True)
    dummydf=pd.concat([dummydf,dummy],axis=1)

print(data.dtypes) 
print(dummydf)


#Merge the dummy and dataset
data1=data
data1=pd.concat([data1,dummydf],axis=1)
data1.head()

#Declare the dependent variable and create your independent and dependent datasets
dep='admit'
obj1=data1.dtypes == py.object
print(obj1)
X=data1.drop(data1.columns[obj1],axis=1)
X=X.drop([dep],axis=1)


#To understand about types of different dataframe column
df=data
df.columns.to_series().groupby(df.dtypes).groups

#Appending V to columns in order to avoid numeric ccolumns names
X.columns='V_'+X.columns
Y=data1[dep]
X.columns


#Split into train and test 
X_train,X_test,Y_train,Y_test=sklearn.cross_validation.train_test_split(X,Y,test_size=0.20,random_state=5)
print('Train Data Size-',X_train.shape[0], '\n')
print('Test Data Size-',X_test.shape[0], '\n')


#Run Model
modLR=sm.Logit(pd.DataFrame(Y_train),X_train,family=sm.families.Binomial())
result=modLR.fit()
result.summary()

result.params
#Odds ratio
py.exp(result.params)

#Predict and get the measure for train and test
cutoff=0.5

pred_train=pd.DataFrame(result.predict(X_train))
predLR=pred_train.loc[:,0].map(lambda x: 1 if x > cutoff else 0)
ResLR=measures('Logistic',Y_train,predLR,cutoff)
ResLR

pred_test=pd.DataFrame(result.predict(X_test))
tpredLR=pred_test.loc[:,0].map(lambda x: 1 if x > cutoff else 0)
tResLR=measures('Logistic',Y_test,tpredLR,cutoff)
tResLR

