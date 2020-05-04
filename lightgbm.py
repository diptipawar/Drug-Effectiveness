import pandas as pd 
df = pd.read_csv("clean-train-data.csv") 
df.head(5)

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

params = {
    'boosting_type': 'gbdt', 
    'objective': 'regression',
     #'metric':{'l2','l1'},
'max_depth':-1,
     'num_leaves':13,
'learning_rate':0.01,
'feature_fraction':0.9,
'bagging_fraction':0.8,
'bagging_freq':5,
'boost_from_average': 'False',
'min_sum_hessian_in_leaf':10,
'verbosity':1,
'verbose':0
}
X = df.iloc[:, 1:9].values
y = df.iloc[:, 9].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
dtrain = lgb.Dataset(X, y)
#dtrain = xgb.DMatrix(X_train, y_train)
dtest = lgb.Dataset(X_test, y_test)

#watchlist = [(dtrain, 'train'), (dtest, 'test')]
#watchlist = [(dtrain, 'train')]

gbm = lgb.train(params, dtrain, valid_sets=dtrain,  num_boost_round=200000,early_stopping_rounds = 3000, verbose_eval = True)
#gbm = lgb.train(params, dtrain, num_boost_round=90000,early_stopping_rounds = 75, verbose_eval = True)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


df = pd.read_csv('clean-test-data.csv')
test = df.iloc[:, 1:9].values

base_score = gbm.predict(test,num_iteration=gbm.best_iteration)

my_submission = pd.DataFrame({'patient_id': df.patient_id, 'base_score': base_score})
my_submission.to_csv('gbm1.csv', index=False)

#import pickle
#save model to file
#pickle.dump(xgb_model, open("submission-xgboost-full-train.dat", "wb"))
#print("Saved model to: boost.joblib.dat")

# load model from file
#loaded_model = pickle.load(open("submission-xgboost-full-train.dat", "rb"))

