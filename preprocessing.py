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

# 필요없는 feature를 지우고 했는데, 코드내에서 지워야 할 수 도 있을것 같네요
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn3.csv')

###############################################
########## Data Exploration ###################
# Print dataset statistical df
# numeric feature들의 이상치를 확인할 수 있음
print(df.describe())

print('infomation')
print(df.info())

print('------Data Shape')
print(df.shape)

# Create a function to generate boxplots
plots = {1 : [111], 2:[121,122], 3:[131,132,133],4:[221,222,223,224],5:[231,232,233,234,235],
6 : [231,232,233,234,235,236]}

def count_boxplot(x,y, df):
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    plt.figure(figsize = (7*columns, 7*rows))

    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.boxplot(x=x, y=j, data=df[[x,j]], linewidth=1)
        ax.set_title(j)

    return plt.show()

# Generate boxplots
count_boxplot("Churn", ["tenure", "MonthlyCharges"], df)

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
countplot("Churn", ['SeniorCitizen', 'Contract', 'Partner', 'Dependents'], df)


###################################
########################################################################
# outlier 쳐내기

# Numeric outlier 쳐내기
# 양의 값만 있으므로 0 이하 값 쳐냄
df = df[df['MonthlyCharges'] > 0]
df = df[df['tenure'] > 0]

# boxplot을 이용
# show boxplot
def boxplot(col):
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df[[col]], color='red')
    # plt.show()

# show boxplot
# boxplot('tenure')
# boxplot('MonthlyCharges')

#outlier 지워주는 함수
def outliers_iqr(data,col):
    q1, q3 = np.percentile(data[col],[25,75])
    iqr = q3-q1
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)

    outlier_idx = data[col][(data[col] < lower_bound) | (data[col] > upper_bound)].index
    return outlier_idx

tenure_outlier_index = outliers_iqr(df, 'tenure')
Charges_outlier_index = outliers_iqr(df,'MonthlyCharges')


print(df.loc[tenure_outlier_index, 'tenure']) #2개
print(df.loc[Charges_outlier_index, 'MonthlyCharges']) # 0개

print(df.shape)
df = df.drop(tenure_outlier_index, axis=0) # 2개 사라짐
print(df.shape)

# Drop the rows with missing values.
df = df.dropna()

##################################################
# 실제 의미 있는 값들만 저장하면 됨
# 다른값들은 전부 NaN 바뀜
# 그냥 인코딩을 사용하면 이상한 값도 인코딩 됨.
# 우선 Categorical Outlier부터 쳐내기

# label encoding
gender_mapper = {'Female':'Female','Male':'Male'}
df['gender'] = df['gender'].map(gender_mapper)
df['gender'] = df['gender'].map({'Female' : 1, 'Male' : 0})

def label_encoding(features,df):
    for i in features:
        df[i] = df[i].map({'Yes' : 1, 'No':0})
    return
label_encoding(['Partner', 'Dependents','PhoneService','Churn'],df)


bundle_mapper = {'Yes':'Yes', 'No':'No','No internet service' : 'No internet service'}
zero_one_mapper = {0:0,1:1}
contract_mapper = {'Month-to-month' : 'Month-to-month', 'One year':'One year', 'Two year':'Two year'}

df['SeniorCitizen'] = df['SeniorCitizen'].map(zero_one_mapper)
df['Contract'] = df['Contract'].map(contract_mapper)

# clean dirty data
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
# # NaN 쳐내기
df =  df.dropna()
print(df.info())

#########################################################################
#### Scaling feature
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
