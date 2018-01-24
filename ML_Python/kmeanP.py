##K Means Clustering

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Get the Data
import os
os.getcwd()
os.chdir("C:\\Users\\abhishek.b.jaiswal\\Desktop\\DataScience\\sem 2\\BD 3\\codes")
os.getcwd()
df = pd.read_csv('College_Data',index_col=0)
print("Data shape:", df.shape)
print(df.head())
print(df.dtypes)
print(df.info())
print(df.describe())

#EDA
#Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column.
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)

#Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)

#Create a stacked histogram showing Out of State Tuition based on the Private column. Try doing this using sns.FacetGrid. If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist').
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

#Create a similar histogram for the Grad.Rate column.
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


df[df['Grad.Rate'] > 100]
df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate'] > 100]

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

#K Means Cluster Creation
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))
kmeans.cluster_centers_


#Evaluation
#Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
    

df['Cluster'] = df['Private'].apply(converter)
df.head()
#Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))
    
