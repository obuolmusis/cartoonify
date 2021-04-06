#!/usr/bin/env python
# coding: utf-8

# ---
# # Below I provide a solution to the problem.
# ## I used 75% of the given data for model training and the rest 25% was used for testing.

# In[26]:


# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import tree


# In[27]:


# Read the SeoulBikeData_mod.csv file to dataframe bike
bike = pd.read_csv("SeoulBikeData_mod.csv")
bike.describe()


# In[28]:


# Histogram of rented bike count
bike.hist('Rented Bike Count')
plt.xlabel("Rented Bike Count", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.show()


# In[29]:


# Violin plots of rented bike count
sns.violinplot(x='Month', y='Rented Bike Count', data=bike, inner='quartile')
plt.title('Rented Bike Count per Month')
plt.show()
sns.violinplot(x='Hour', y='Rented Bike Count', data=bike, inner='quartile')
plt.title('Rented Bike Count per Hour')
plt.show()


# In[30]:


# Extract values from the dataset
X=bike.drop(['Rented Bike Count'], axis=1)
y=bike['Rented Bike Count']

# Split the samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the train data
regr.fit(X_train, y_train)

# Use the model to predict the test data
y_pred = regr.predict(X_test)

# Form the residual
resid = y_test - y_pred

RMSE = np.sqrt(np.mean(resid**2))
MAE = np.mean(np.abs(resid))
R2 = r2_score(y_test, y_pred)
print('The value of the Root Mean Squared Error is', RMSE)
print('The value of the Mean Absolute Error is', MAE)
print('The value of the Rsquared is', R2)


# In[31]:


# plot scatter Rented Bike Count - Residual
plt.scatter(y_test, resid, s=30, c = X_test['Month'], cmap = 'jet')
plt.xlabel("Rented Bike Count", fontsize="14")
plt.ylabel("Residual", fontsize="14")
plt.title("linear regression model residual map", fontsize="16")
plt.show()


# In[32]:


#plot histogram y_test - y_pred
plt.hist(y_test, color='red', density=True, label="test values", edgecolor = 'k')
plt.hist(y_pred, color='green', density=True, label="predicted values", edgecolor = 'k')
plt.xlabel('Rented Bike Count')
plt.ylabel('Density')
plt.legend(fontsize=12)
plt.show()


# In[33]:


# plot scatter Month - Residual
plt.scatter(X_test['Month'], resid , s=40, c=abs(resid) ,cmap='plasma', alpha=1)
plt.colorbar(label='the absolute value of residual')
plt.ylabel('Residual')
plt.xlabel('Month')
plt.show()


# In[34]:


# plot scatter Month - Residual
plt.scatter(X_test['Hour'], resid, s=30, c=abs(resid) ,cmap='hsv', alpha=0.8)
plt.colorbar(label='the absolute value of residual')
plt.ylabel('Residual')
plt.xlabel('Hour')
plt.show()


# Optional part

# In[35]:


# Fit regression model
regr_1 = tree.DecisionTreeClassifier(max_depth=6)
regr_1.fit(X_train, y_train)

y_pred1 = regr_1.predict(X_test)


# In[36]:


# Form the residual
resid1 = y_test - y_pred1

RMSE = np.sqrt(np.mean(resid1**2))
MAE = np.mean(np.abs(resid1))
R2 = r2_score(y_test, y_pred1)
print('The value of the Root Mean Squared Error is', RMSE)
print('The value of the Mean Absolute Error is', MAE)
print('The value of the Rsquared is', R2)


# In[37]:


#plot histogram y_test - y_pred
plt.hist(y_test, color='red', density=True, label="test values", edgecolor = 'k')
plt.hist(y_pred1, color='green', density=True, label="predicted values", edgecolor = 'k')
plt.xlabel('Rented Bike Count')
plt.ylabel('Frequency')
plt.title('Alternate graph from DecisionTreeRegressor')
plt.legend(fontsize=12)
plt.show()

