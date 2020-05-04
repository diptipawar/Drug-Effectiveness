#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor


# In[2]:


import pandas as pd 
df = pd.read_csv("clean-train-data.csv") 
df.head(5)


# In[6]:


X = df.iloc[:, 1:9].values
y = df.iloc[:, 9].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=0)


# In[10]:


model = RandomForestRegressor(n_estimators=6000, 
                               bootstrap = True,
                               max_features = 'auto',
				max_depth=10)


# In[11]:


#model.fit(X_train,y_train)
model.fit(X,y)


# In[9]:


from sklearn.metrics import mean_squared_error
import numpy as np
rmse_train=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
print('RMSE: %4f' % rmse_train)


# In[10]:


rmse_test=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
print('RMSE: %4f' % rmse_test)


# In[11]:


y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
# evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

my_submission = pd.DataFrame({'y_test': y_test,'y_pred':y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('validate_check.csv', index=False)

df = pd.read_csv('clean-test-data.csv')
test = df.iloc[:, 1:9].values

base_score = model.predict(test)

my_submission = pd.DataFrame({'Id': df.patient_id, 'basescore': base_score})
my_submission.to_csv('final_submission_rf.csv', index=False)




