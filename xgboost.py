#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import scipy.sparse as sp
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.trial import TrialState
from optuna.integration import XGBoostPruningCallback
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet


# In[2]:


# import df.csv

df = pd.read_csv('df.csv')


# In[3]:



# Convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df.sort_values('Date', inplace=True)

# Remove rows where the 'Injury' column has NA values
df = df.dropna(subset=['Injury'])

# Get unique dates
unique_dates = df['Date'].unique()

# Decide how many dates to include in each set
train_dates = int(len(unique_dates) * 0.8)

# Find the index of the last training date
last_train_idx = df[df['Date'] == unique_dates[train_dates]].index[-1]


train = df.iloc[:last_train_idx + 1]
test = df.iloc[last_train_idx + 1:]


# In[4]:


# Identify categorical columns excluding the date column
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

# Initialize OneHotEncoder
ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')

# Fit the OneHotEncoder on the training data and transform both train and test data
ohe.fit(train[cat_cols])
one_hot_encoded_train = ohe.transform(train[cat_cols])
one_hot_encoded_test = ohe.transform(test[cat_cols])

# Extract numeric data
numeric_data_train = train.drop(cat_cols + ['Date', 'Injury'], axis=1).values
numeric_data_test = test.drop(cat_cols + ['Date', 'Injury'], axis=1).values


# Combine sparse and dense data
final_data_train = sp.hstack((one_hot_encoded_train, numeric_data_train)).tocsr()
final_data_test = sp.hstack((one_hot_encoded_test, numeric_data_test)).tocsr()



# Set up for cross-validation
X = final_data_train
y = train["Injury"].values  

X_test = final_data_test
y_test = test["Injury"].values

# Initialize a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)


# In[5]:


rows, cols = X.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {cols}")


# In[6]:


def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0.0, 1.0),
        'objective': 'reg:squarederror',
        #'tree_method': 'gpu_hist',  # use GPU-based algorithm
        #'gpu_id': 1  # ID of the GPU to use
    }

    num_boost_round = trial.suggest_int('num_boost_round', 50, 500)
    early_stopping_rounds = 50  # Stop if performance hasn't improved for 50 rounds

    mse_scores = []
    for train_index, val_index in tscv.split(X):
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y[train_index], y[val_index]

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
        dval = xgb.DMatrix(X_val_cv, label=y_val_cv)

        # Train the model
        model = xgb.train(
            params, 
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'eval')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        # Predict the validation set
        y_pred = model.predict(dval)

        mse = mean_squared_error(y_val_cv, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores)

# Create a study that uses SQLite storage
study = optuna.create_study(direction='minimize', storage='sqlite:///example.db', load_if_exists=True, study_name='no-name-286bbea7-b86c-450d-af18-d85fcadd081f')  


# In[7]:



# Optimize the study
study.optimize(objective, n_trials=100, catch=(Exception,))


# In[8]:


# Extract the best parameters
best_params = study.best_params




early_stopping_rounds = 50 # Stop if performance hasn't improved for 50 rounds

# Prepare the DMatrix format for the entire training data and test data
dtrain_full = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test, label=y_test)  

# Train the model with the best parameters on the training data
final_model = xgb.train(
    best_params, 
    dtrain_full, 
    early_stopping_rounds=early_stopping_rounds,
    evals=[(dtrain_full, 'train')],
    verbose_eval=True 
    
)

# Make predictions on the test data
predictions_test = final_model.predict(dtest)

# Calculate mean squared error on the test data
mse_test = mean_squared_error(y_test, predictions_test)

print("Mean squared error on the test data: ", mse_test)



# In[9]:


best_params


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Injury'], marker='o', linestyle=':', color='blue')

# Calculating the cutoff date at 80% of the data
cutoff_percentage = 0.8
cutoff_index = int(len(df) * cutoff_percentage)
cutoff_date = df['Date'].iloc[cutoff_index]

# Adding the vertical red dotted line at the cutoff date
plt.axvline(x=cutoff_date, color='red', linestyle='--')

plt.title('Injury Over Time')
plt.xlabel('Date')
plt.ylabel('Injury')
plt.show()


# In[10]:




# plot the feature importance
xgb.plot_importance(final_model)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()


# In[21]:


# Get one-hot encoded feature names
ohe_feature_names = ohe.get_feature_names_out(cat_cols)

# Get numeric feature names
numeric_feature_names = train.drop(cat_cols + ['Date'], axis=1).columns

# Combine both lists
all_feature_names = np.concatenate([ohe_feature_names, numeric_feature_names])


print(all_feature_names[208])
print(all_feature_names[264])
print(all_feature_names[254])
print(all_feature_names[209])
print(all_feature_names[210])

