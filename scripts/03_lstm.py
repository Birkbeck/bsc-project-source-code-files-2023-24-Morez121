# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import utils

# global variable
TRAIN_TXT_PATH = "./data/KDDTrain+.txt"
METADATA_PATH = "./data/KDDTrain+.arff"
TEST_PATH = "./data/KDDTest+.txt"
TEST_EXC_21_PATH = "./data/KDDTest-21.txt"
SEED = 111
LABEL = "class"
EPOCHS = 5 # no. of iteration over data


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
# Reshape input data to add timestep dimension
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# %%
model = tf.keras.Sequential([
    LSTM(units=64, activation='relu', input_shape=(1, X_train.shape[2]), 
         kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4), 
         bias_regularizer=tf.keras.regularizers.L2(1e-4),
         activity_regularizer=tf.keras.regularizers.L2(1e-5),
         return_sequences=True),
    Dropout(0.4),
    LSTM(units=128, activation='relu', 
         kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4), 
         bias_regularizer=tf.keras.regularizers.L2(1e-4),
         activity_regularizer=tf.keras.regularizers.L2(1e-5),
         return_sequences=True),
    Dropout(0.4),
    LSTM(units=512, activation='relu', 
         kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4), 
         bias_regularizer=tf.keras.regularizers.L2(1e-4),
         activity_regularizer=tf.keras.regularizers.L2(1e-5),
         return_sequences=False),
    Dropout(0.4),
    Dense(units=128, activation='relu', 
          kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4), 
          bias_regularizer=tf.keras.regularizers.L2(1e-4),
          activity_regularizer=tf.keras.regularizers.L2(1e-5)),
    Dropout(0.4),
    Dense(units=1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, verbose=1)


# %%
# load test data
df_test = pd.read_csv(TEST_PATH)
display(df_test)
df_test = pre_pre_process_data(df_test)

X_test = df_test[FEATURES]
y_test = df_test[LABEL]

# Reshape input data to add timestep dimension
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

y_test_pred = model.predict(X_test)
y_test_pred = list(map(lambda x:1 if x>=0.5 else 0, y_test_pred[:, 0]))


# %%
print("Neural-Network MODEL:")
print("AUC: ", roc_auc_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, digits=4), end="---\n")

# %%



