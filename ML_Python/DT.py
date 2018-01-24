#Load Packages and set Directory
import pandas as pd
import numpy as py
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#Load data
import os
data = pd.read_csv('C:/Users/abhishek.b.jaiswal/Desktop/DataScience/sem 2/BD 3/binary.csv')
print("Data shape:", data.shape)
print(data.head())
print(data.dtypes)


data['rank'] = data['rank'].astype(str)

dep = 'admit'

obj = data.dtypes == py.object
obj[dep] = False
dummydf = pd.DataFrame()

for i in data.columns[obj]:
    dummy = pd.get_dummies(data[i], drop_first=True)
    dummydf = pd.concat([dummydf, dummy], axis = 1)
    

#Split into train and test
data1 = data
data1 = pd.concat([data1,dummydf], axis = 1)
obj1 = data1.dtypes == py.object

X = data1.drop(data1.columns[obj1], axis = 1)
X = X.drop([dep], axis = 1)

X.columns = 'V_' + X.columns
print("X_col \n", X.columns)
Y = data1[dep]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state =5)

print('Train Data Size - ', X_train.shape[0], '\n')
print('Test  Data Size - ', X_test.shape[0])    


#Model creation
modCART = DecisionTreeClassifier()
param_grid = {'max_depth': py.arange(3,10)}
gridS = GridSearchCV(modCART, param_grid)
gridS.fit(X_train, Y_train)
tree_preds = gridS.predict_proba(X_test)[:,1]
tree_performance = roc_auc_score(Y_test, tree_preds)

print('DecisionTree: Area under the ROC Curve = {}'.format(tree_performance))

gridS.best_params_
gridS.grid_scores_

modCART = DecisionTreeClassifier(max_depth=5)
modCART.fit(X_train, Y_train)


from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
%matplotlib inline
import pydotplus
import pydot
dot_data = StringIO()
export_graphviz(modCART, out_file=dot_data,
               filled = True, rounded=True,
               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
