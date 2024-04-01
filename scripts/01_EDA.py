# %%
# EDA: Exploratory Data Analysis

# imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import utils

# global variable
TRAIN_TXT_PATH = "./data/KDDTrain+.txt"
METADATA_PATH = "./data/KDDTrain+.arff"
TEST_PATH = "./data/KDDTest+.txt"
TEST_EXC_21_PATH = "./data/KDDTest-21.txt"
SEED = 111
LABEL = "class"


# %%
# load data
df = pd.read_csv(TRAIN_TXT_PATH)
df

# %%
df.columns = utils.get_col_names(METADATA_PATH)
df



# %%
# check for null values
df.isna().sum().sum()

# %%
# check data types
df.info()
# check numeric cols
utils.get_numeric_cols(df)
# check numeric cols
utils.get_numeric_cols(df)



# %%
# get generic stats
df.describe().T

# %%
# Plot bar charts for the label
plt.figure(figsize=(10,6), dpi=160)
df[LABEL].value_counts().plot(kind='bar')
plt.title('Bar Chart for Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.show()


# %%
# Plot bar charts for the label (after converting to binary)
df = utils.convert_label_to_binary(df, LABEL)

plt.figure(figsize=(10,6), dpi=160)
df[LABEL].value_counts().plot(kind='bar')
plt.title('Bar Chart for Labels (after converting to binary)')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.show()

# %%
# plot correlation heatmap

df_num = utils.get_numeric_cols(df)
corr = df_num.corr()

plt.figure(figsize=(16, 12), dpi=200)
sns.heatmap(corr, linewidths=0.1, vmax=1.0, square=False, linecolor='white')
plt.title("Correlation Heatmap")
plt.show()



# %%
# # plot distribution of each feature for each class

# nrow=10
# ncol=4
# f, axes = plt.subplots(nrow, ncol, figsize=(20,20))
# i = j = 0
# for col in df_num.columns:
#     sns.distplot(df_num[df_num[LABEL]==1][col] , ax=axes[i, j], hist=False, label="Attack")
#     sns.distplot(df_num[df_num[LABEL]==0][col] , ax=axes[i, j], hist=False, label="Normal")
#     plt.legend()
#     if j<ncol:
#         j += 1
#     if j==ncol:
#         i += 1
#         j = 0

# plt.tight_layout()
# plt.show()


# %%
# plot pie chart for categorical features
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    fig = px.pie(df, names=col)
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    # add title
    fig.update_layout(title=f"Pie chart for {col}")
    fig.show()



# %%



