# %%
# Models: modeling with different algorithms

# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, auc, roc_auc_score

import utils

# global variable
TRAIN_TXT_PATH = "./data/KDDTrain+.txt"
METADATA_PATH = "./data/KDDTrain+.arff"
TEST_PATH = "./data/KDDTest+.txt"
TEST_EXC_21_PATH = "./data/KDDTest-21.txt"
SEED = 111
LABEL = "class"


# %%
# load and process data
df = pd.read_csv(TRAIN_TXT_PATH)

def pre_pre_process_data(df):
    df.columns = utils.get_col_names(METADATA_PATH)
    df = utils.convert_label_to_binary(df, LABEL)
    df = utils.get_numeric_cols(df)
    return df

df = pre_pre_process_data(df)
df



# %%
# X-y split (features & label) - avoid data-leakage
X = df.drop(columns=["class", "level"], axis=1)
FEATURES = X.columns

y = df[LABEL]

# train-val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=SEED)
print(f"{X_train.shape=} {y_train.shape=} {X_val.shape=} {y_val.shape=}")
display(X_train)
display(y_train)


# %%
# 1: model with naive bayes
from sklearn.naive_bayes import BernoulliNB
nb_model = BernoulliNB().fit(X_train, y_train)
y_val_pred = nb_model.predict(X_val)
print(classification_report(y_val, y_val_pred,  digits=4))
print("AUC: ", roc_auc_score(y_val, y_val_pred))

# %%
# 2: model with logistic regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression().fit(X_train, y_train)
y_val_pred = lr_model.predict(X_val)
print(classification_report(y_val, y_val_pred,  digits=4))
print("AUC: ", roc_auc_score(y_val, y_val_pred))


# %%
# 3: model with KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier().fit(X_train, y_train)
y_val_pred = knn_model.predict(X_val)
print(classification_report(y_val, y_val_pred,  digits=4))
print("AUC: ", roc_auc_score(y_val, y_val_pred))



# %%
# 4: model with SVM
# from sklearn.svm import SVC
# svm_model = SVC().fit(X_train, y_train)
# y_val_pred = svm_model.predict(X_val)
# print(classification_report(y_val, y_val_pred,  digits=4))

# #               precision    recall  f1-score   support

# #            0     0.5343    0.9999    0.6965     20182
# #            1     0.9231    0.0014    0.0027     17610

# #     accuracy                         0.5346     37792
# #    macro avg     0.7287    0.5006    0.3496     37792
# # weighted avg     0.7155    0.5346    0.3732     37792


# %%
# model with xgboost (ensemble)
from xgboost import XGBClassifier

# fit (train) model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# test model
y_val_pred = xgb_model.predict(X_val)

# evaluate model
print(classification_report(y_val, y_val_pred,  digits=4))
print("AUC: ", roc_auc_score(y_val, y_val_pred))


# %%
# # model with ensemble majority voting
# from sklearn.ensemble import VotingClassifier

# ensemble_model = VotingClassifier(estimators=[
#     ('xgb', xgb_model),
#     ('lr', lr_model),
#     ('knn', knn_model),
# ], voting='soft')
# ensemble_model.fit(X_train, y_train)

# y_val_pred = ensemble_model.predict(X_val)
# print(classification_report(y_val, y_val_pred,  digits=4))
# print("AUC: ", roc_auc_score(y_val, y_val_pred))

# %%
# load test data
df_test = pd.read_csv(TEST_PATH)
display(df_test)
df_test = pre_pre_process_data(df_test)

X_test = df_test[FEATURES]
y_test = df_test[LABEL]

# %%
# evaluate models on the test set
print("Naive-Bayes MODEL:")
y_test_pred = nb_model.predict(X_test)
print("AUC: ", roc_auc_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, digits=4), end="---\n")

print("Logistic-Regression MODEL:")
y_test_pred = lr_model.predict(X_test)
print("AUC: ", roc_auc_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, digits=4), end="---\n")

print("KNN MODEL:")
y_test_pred = knn_model.predict(X_test)
print("AUC: ", roc_auc_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, digits=4), end="---\n")

# print("SVM MODEL:")
# y_test_pred = svm_model.predict(X_test)
# print("AUC: ", roc_auc_score(y_test, y_test_pred))
# print(classification_report(y_test, y_test_pred, digits=4), end="---\n")

print("XGB MODEL:")
y_test_pred = xgb_model.predict(X_test)
print("AUC: ", roc_auc_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, digits=4), end="---\n")


# %%
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 10, 100, 1000],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.2, 0.5, 0.7, 1]
}

# Create the XGBoost model object
xgb_model = xgb.XGBClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# %%
print("XGB MODEL:")
xgb_model = xgb.XGBClassifier(**grid_search.best_params_)
xgb_model.fit(X_train, y_train)
y_test_pred = xgb_model.predict(X_test)
print("AUC: ", roc_auc_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, digits=4), end="---\n")

# %%
m = xgb.XGBClassifier()
m.get_params()

# %%



