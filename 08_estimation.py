import pandas as pd
import importlib
importlib.reload(utils)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import utils

# Load data set
df = pd.read_csv('data/final_data_set.csv', sep=';', encoding='utf-8')

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df[df.columns[1:]], df['Final'], test_size=0.2, random_state=123)

# Get numerical and categorical indices
num_vars = [list(X_train.columns).index(col) for col in X_train.columns if col not in ['Country', 'Club', 'First comp in class']]
cat_vars = list(set(range(0, X_train.shape[1])) - set(num_vars))

# MLP with grid search
mlp_pipe = utils.make_pipeline(MLPClassifier(random_state=123, max_iter=50), num_vars, cat_vars)
mlp_grid = {'model__hidden_layer_sizes':[20, 50, 100, 200], 'model__activation':['relu', 'tanh', 'logistic'], 'model__alpha':[0, 0.0001, 0.001, 0.01, 0.1]}
mlp_model = GridSearchCV(mlp_pipe, mlp_grid, scoring='roc_auc', n_jobs=-1, cv=3, verbose=4)
mlp_model.fit(X_train, y_train)
pd.DataFrame(mlp_model.cv_results_).to_excel('models/MLP_grid_results.xlsx', index=False) 
roc_auc_score(y_test, mlp_model.predict_proba(X_test)[:, 1]) # 0.831297120675842

# MLP with grid search (no cats) - except 'First comp in class'
mlp_pipe_no_cats = utils.make_pipeline(MLPClassifier(random_state=123, max_iter=50), list(range(X_train.shape[1] - 3)))
mlp_grid_no_cats = {'model__hidden_layer_sizes':[1, 2, 3, 5, 10, 20], 'model__activation':['relu', 'tanh', 'logistic'], 'model__alpha':[0, 0.0001, 0.001, 0.01, 0.1]}
mlp_model_no_cats = GridSearchCV(mlp_pipe_no_cats, mlp_grid_no_cats, scoring='roc_auc', n_jobs=-1, cv=3, verbose=4)
mlp_model_no_cats.fit(X_train[X_train.columns[num_vars + [4]]], y_train)
pd.DataFrame(mlp_model_no_cats.cv_results_).to_excel('models/MLP_grid_results_no_cats.xlsx', index=False)
roc_auc_score(y_test, mlp_model_no_cats.predict_proba(X_test[X_train.columns[num_vars + [4]]])[:, 1]) #0.8288864606250598

# MLP with grid search (naive)
mlp_pipe_naive = utils.make_pipeline(MLPClassifier(random_state=123, max_iter=50), [0, 1])
mlp_grid_naive = {'model__hidden_layer_sizes':[1, 2, 3, 5, 10, 20], 'model__activation':['relu', 'tanh', 'logistic'], 'model__alpha':[0, 0.0001, 0.001, 0.01, 0.1]}
mlp_model_naive = GridSearchCV(mlp_pipe_naive, mlp_grid_naive, scoring='roc_auc', n_jobs=-1, cv=3, verbose=4)
mlp_model_naive.fit(X_train[['Points before', 'Finals before']], y_train)
pd.DataFrame(mlp_model_naive.cv_results_).to_excel('models/MLP_grid_results_naive.xlsx', index=False)
roc_auc_score(y_test, mlp_model_naive.predict_proba(X_test[['Points before', 'Finals before']])[:, 1]) # 0.7518153747436269

# Logreg with grid search
log_reg_pipe = utils.make_pipeline(LogisticRegression(random_state=123, max_iter=500), num_vars, cat_vars)
log_reg_grid = [{'model__penalty':['l2'], 'model__C':[0.001, 0.01, 0.1, 0.5, 1, 5, 10]}, {'model__penalty':['l1'], 'model__solver' : ['liblinear'], 'model__C':[0.001, 0.01, 0.1, 0.5, 1, 5, 10]}, {'model__penalty':[None]}]
log_reg_model = GridSearchCV(log_reg_pipe, log_reg_grid, scoring='roc_auc', n_jobs=-1, cv=3, verbose=4)
log_reg_model.fit(X_train, y_train)
pd.DataFrame(log_reg_model.cv_results_).to_excel('models/LR_grid_results.xlsx', index=False) 
roc_auc_score(y_test, log_reg_model.predict_proba(X_test)[:, 1]) # 0.8267652586088822

# Logreg with polynomials
log_reg_poly_pipe = utils.make_pipeline(LogisticRegression(random_state=123, max_iter=500), num_vars, cat_vars, poly=True)
log_reg_poly_grid = [{'model__penalty':['l2'], 'model__C':[0.001, 0.01, 0.1, 0.5, 1, 5, 10]}, {'model__penalty':['l1'], 'model__solver' : ['liblinear'], 'model__C':[0.001, 0.01, 0.1, 0.5, 1, 5, 10]}, {'model__penalty':[None]}]
log_reg_poly_model = GridSearchCV(log_reg_poly_pipe, log_reg_poly_grid, scoring='roc_auc', n_jobs=-1, cv=3, verbose=4)
log_reg_poly_model.fit(X_train, y_train)
pd.DataFrame(log_reg_poly_model.cv_results_).to_excel('models/LR_poly_grid_results.xlsx', index=False) 
roc_auc_score(y_test, log_reg_poly_model.predict_proba(X_test)[:, 1]) # 0.8295947077048644

# Best MLP model - base
mlp_best = utils.make_pipeline(MLPClassifier(random_state=123, max_iter=500, hidden_layer_sizes=100, alpha=0.01, activation='tanh'), num_vars, cat_vars)
mlp_best.fit(X_train, y_train)
roc_auc_score(y_test, mlp_best.predict_proba(X_test)[:, 1]) # 0.8326008344910798

# Best MLP model - no cats
mlp_best_no_cats = utils.make_pipeline(MLPClassifier(random_state=123, max_iter=500, hidden_layer_sizes=20, alpha=0, activation='tanh'), list(range(X_train.shape[1] - 3)))
mlp_best_no_cats.fit(X_train[X_train.columns[num_vars + [5]]], y_train)
roc_auc_score(y_test, mlp_best_no_cats.predict_proba(X_test[X_train.columns[num_vars + [5]]])[:, 1]) # 0.8292492369696329

# Best MLP model - naive
mlp_best_naive = utils.make_pipeline(MLPClassifier(random_state=123, max_iter=500, hidden_layer_sizes=20, alpha=0, activation='relu'), [0, 1, 2])
mlp_best_naive.fit(X_train[['Points before', 'Finals before', 'N']], y_train)
roc_auc_score(y_test, mlp_best_naive.predict_proba(X_test[['Points before', 'Finals before', 'N']])[:, 1]) # 0.7560819704655737

# # Sklearn log reg
# model = LogisticRegression(n_jobs=-1, max_iter=1000)
# model.fit(X_train_new, y_train)
# model.score(X_train_new, y_train)
# model.score(X_test_new, y_test)
# roc_auc_score(y_test, model.predict_proba(X_test_new)[:, 1])
# roc_auc_score(y_train, model.predict_proba(X_train_new)[:, 1])

# # Statsmodels log reg
# log_reg = sm.Logit(y_train, sm.add_constant(X_train_new)).fit(cov_type='hc3')
# log_reg.summary()

# # MLP
# model_nn = MLPClassifier(activation='relu', random_state=123)
# model_nn.fit(X_train_new, y_train)
# roc_auc_score(y_test, model_nn.predict_proba(X_test_new)[:, 1])
# roc_auc_score(y_train, model_nn.predict_proba(X_train_new)[:, 1])

# # Random Forest
# model_RF = RandomForestClassifier(random_state=123, max_depth=10)
# model_RF.fit(X_train_new, y_train)
# roc_auc_score(y_test, model_RF.predict_proba(X_test_new)[:, 1])
# roc_auc_score(y_train, model_RF.predict_proba(X_train_new)[:, 1])

# # Naive MLP
# model_nn = MLPClassifier(hidden_layer_sizes=1, activation='relu', random_state=123)
# model_nn.fit(X_train_new[['Points before', 'Finals before']], y_train)
# roc_auc_score(y_test, model_nn.predict_proba(X_test_new[['Points before', 'Finals before']])[:, 1])
# roc_auc_score(y_train, model_nn.predict_proba(X_train_new[['Points before', 'Finals before']])[:, 1])

# # MLP no club and country
# model_nn = MLPClassifier(hidden_layer_sizes=8, activation='relu', random_state=123)
# model_nn.fit(X_train_new[num_vars], y_train)
# roc_auc_score(y_test, model_nn.predict_proba(X_test_new[num_vars])[:, 1])
# roc_auc_score(y_train, model_nn.predict_proba(X_train_new[num_vars])[:, 1])
