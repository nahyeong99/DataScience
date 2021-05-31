# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from seaborn.categorical import countplot
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# read CSV
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

###############################################
########## Data Exploration ###################
###### Print dataset statistical df
print("######################")
print("Original data")
print(df.shape)
print(df.describe())

print('infomation')
print(df.info())

print('------Data Shape')
print(df.shape)

print(df.columns)

# Display a frequency distribution for churn
plt.figure(figsize=(5,5))
ax = sns.countplot(x=df['Churn'])
plt.show()

# Create a function to generate boxplots
plots = {1 : [111], 2:[121,122], 3:[131,132,133],4:[221,222,223,224],5:[231,232,233,234,235],
6 : [231,232,233,234,235,236]}

# Can check out outliers in numerical features
def count_boxplot(x,y, df):
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    plt.figure(figsize = (7*columns, 7*rows))

    # i : index, j : item
    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.boxplot(x=x, y=j, data=df[[x,j]], linewidth=1)
        ax.set_title(j)

    return plt.show()

# change numeric type
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Generate boxplots
count_boxplot("Churn", ["tenure", "MonthlyCharges", "TotalCharges"], df)

# categorical
def countplot(x,y, df) :
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    plt.figure(figsize = (7*columns, 7*rows))

    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.countplot(x=j, hue=x, data=df)
        ax.set_title(j)

    return plt.show()

# Generate countplots
countplot("Churn", ['SeniorCitizen', 'Contract', 'Partner', 'Dependents', 'PaymentMethod', 'InternetService'], df)

df = df.drop(["MultipleLines", "InternetService", 'OnlineBackup', 'StreamingMovies','PaperlessBilling','PaymentMethod','TotalCharges'], axis=1)

#######################################
# ######## Remove outlier #############
#######################################
# remove Numeric outlier
# Value below 0 because only positive values exist
print("###############################")
print("Before remove numeric outlier")
print(df.shape)

df = df[df['MonthlyCharges'] > 0]
df = df[df['tenure'] > 0]

# show boxplot
def boxplot(col):
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df[[col]], color='red')
    plt.show()

# show boxplot
boxplot('tenure')
boxplot('MonthlyCharges')

# Remove numeric outlier
def outliers_iqr(data,col):
    q1, q3 = np.percentile(data[col],[25,75])
    iqr = q3-q1
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)

    outlier_idx = data[col][(data[col] < lower_bound) | (data[col] > upper_bound)].index
    return outlier_idx

tenure_outlier_index = outliers_iqr(df, 'tenure')
Charges_outlier_index = outliers_iqr(df,'MonthlyCharges')

print("After remove numeric outlier")
print(df.shape)

# print(df.loc[tenure_outlier_index, 'tenure']) # 2
# print(df.loc[Charges_outlier_index, 'MonthlyCharges']) # 0

print(df.shape)
df = df.drop(tenure_outlier_index, axis=0)

print("After remove numeric outlier")
print(df.shape)

# to check removing outlier data
boxplot('tenure')
# Drop the rows with missing values.
df = df.dropna()

##################################################
# Only need to store actual meaningful values
# All other values changed NaN
# If you just use encoding, even strange values will be encoded.
# must clean dirty data
####################################
# label encoding
print("##########################")
print("Before cleaning Data Shape")
print(df.shape)

gender_mapper = {'Female':'Female','Male':'Male'}
df['gender'] = df['gender'].map(gender_mapper)
df['gender'] = df['gender'].map({'Female' : 1, 'Male' : 0})

def label_encoding(features,df):
    for i in features:
        df[i] = df[i].map({'Yes' : 1, 'No':0})
    return
label_encoding(['Partner', 'Dependents','PhoneService','Churn'],df)

##### clean dirty data
# leave [0, 1] and remove dirty data
zero_one_mapper = {0:0,1:1}
# leave ['Month-to-month', 'One year, 'Two year'] and remove dirty data
contract_mapper = {'Month-to-month' : 'Month-to-month', 'One year':'One year', 'Two year':'Two year'}

df['SeniorCitizen'] = df['SeniorCitizen'].map(zero_one_mapper)
df['Contract'] = df['Contract'].map(contract_mapper)

# clean dirty data
# leave ['Yes', 'No, 'No internet service'] and remove dirty data
def clean_bundle_feature(features, df):
    for i in features:
        df[i] = df[i].map({'Yes':'Yes', 'No':'No','No internet service' : 'No internet service'})
    return
clean_bundle_feature(['OnlineSecurity','DeviceProtection','TechSupport','StreamingTV'],df)

# One - Hot Encoding for identified columns.
features_ohe = ['OnlineSecurity','DeviceProtection','TechSupport','StreamingTV', 'Contract']
df = pd.get_dummies(df, columns=features_ohe)
print(df.columns)

# ###############################
# # Remove NaN
df =  df.dropna()
print(df.info())

print("##########################")
print("After Encoding Data Shape")
print(df.shape)


#####################################
#### Scaling feature ###############
####################################
features_scaling = ['tenure', 'MonthlyCharges']
df_features_scaling = pd.DataFrame(df, columns = features_scaling)
df_remaining_features = df.drop(columns=features_scaling)

###### MinMaxScaler
minmax_scaler = preprocessing.MinMaxScaler()
minmax_features = minmax_scaler.fit_transform(df_features_scaling)

# to change DataFrame
df_minmax_features = \
    pd.DataFrame(minmax_features, columns = features_scaling, index = df_remaining_features.index)

df_minmax = pd.concat([df_remaining_features, df_minmax_features], axis=1)
print("MinMaxScaler")
print(df_minmax.head(),'\n')

###### MaxAbsScaler

maxabs_scaler = preprocessing.MaxAbsScaler()
maxabs_features = maxabs_scaler.fit_transform(df_features_scaling)

# to change DataFrame
df_maxabs_features = \
    pd.DataFrame(maxabs_features, columns = features_scaling, index = df_remaining_features.index)

df_maxabs = pd.concat([df_remaining_features, df_maxabs_features], axis=1)                                           
print("MaxAbsScaler")
print(df_maxabs.head(),'\n')

###### Robust Scaler 

robust_scaler = preprocessing.RobustScaler()
robust_features = robust_scaler.fit_transform(df_features_scaling)

# to change DataFrame
df_robust_features = \
    pd.DataFrame(robust_features, columns = features_scaling, index = df_remaining_features.index)

df_robust = pd.concat([df_remaining_features, df_robust_features], axis=1)                                           
print("RobustScaler")
print(df_robust.head(),'\n')

###### Standard Scaler

standard_scaler = preprocessing.StandardScaler()
standard_features = standard_scaler.fit_transform(df_features_scaling)

# to change DataFrame
df_standard_features = \
    pd.DataFrame(standard_features, columns = features_scaling, index = df_remaining_features.index)

df_standard = pd.concat([df_remaining_features, df_standard_features], axis=1)                                           
print("MaxAbsScaler")
print(df_standard.head(),'\n')