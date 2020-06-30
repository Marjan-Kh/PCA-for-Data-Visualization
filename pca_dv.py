# This program applies PCA to the IRIS dataset to understand
# how PCA is used in visualization and
# how dimensionality reduction is achieved using PCA.

#============================================================
# Author: Marjan Khamesian
# Date: June 2020
#============================================================


import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
print('IRIS dataset')
print(df.head())

# ==== Standardize the Data ==== 

from sklearn.preprocessing import StandardScaler

features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

s_df = pd.DataFrame(data = x, columns = features).head()
print(s_df)

# ==== Dimensionality Reduction ==== 

# PCA 
from sklearn.decomposition import PCA

# Keeping the Top 2 Principal Components
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDF = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

print(principalDF.head())

print(df[['target']].head())

# Final DataFrame before plotting the data.

finalDf = pd.concat([principalDF, df[['target']]], axis = 1)
print('Final DataFrame')
print(finalDf.head())

# ==== Visualization ====

# plot different classes

from matplotlib import pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

# ==== Explained Variance ====

pca_exp_var = pca.explained_variance_ratio_
print('Explained Variance:')
print(pca_exp_var)
