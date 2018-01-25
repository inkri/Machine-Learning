#Principal Component Analysis

#Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#The Data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])

df.head()

#PCA Visualization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

#We can also specify how many components we want to keep when creating the PCA object.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)

#Now we can transform this data to its first 2 principal components.
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

#Great! We've reduced 30 dimensions to just 2! Let's plot these two dimensions out!
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


#Interpreting the components
pca.components_
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)

