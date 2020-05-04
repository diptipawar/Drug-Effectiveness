import pandas as pd 
df = pd.read_csv("clean-train-data.csv") 
df.head(5)

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

params = {
    'booster': 'gbtree', 
    'objective': 'reg:linear',
    'subsample': 0.8, 
    'colsample_bytree': 0.85, 
    'eta': 0.01, 
    'max_depth': 6, 
    'seed': 42}

X = df.iloc[:, 1:9].values
y = df.iloc[:, 9].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=0)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

watchlist = [(dtrain, 'train'), (dtest, 'test')]

xgb_model = xgb.train(params, dtrain, 30000, evals = watchlist,
                      early_stopping_rounds = 50, verbose_eval = True)

df = pd.read_csv('clean-test-data.csv')
test = df.iloc[:, 1:9].values

base_score = xgb_model.predict(xgb.DMatrix(test))

my_submission = pd.DataFrame({'patient_id': df.patient_id, 'base_score': base_score})
my_submission.to_csv('submission4-xgbbost-30000_wt6_05_3.csv', index=False)

import pickle
#save model to file
pickle.dump(xgb_model, open("boost.pickle_30000_1_6_05_3.dat", "wb"))
print("Saved model to: boost.joblib.dat")

# load model from file
loaded_model = pickle.load(open("boost.pickle_30000_1_6_05_3.dat", "rb"))
