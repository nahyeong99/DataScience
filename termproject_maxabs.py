# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from seaborn.categorical import countplot
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('C:/Users/seoyo/PycharmProjects/pythonProject/WA_Fn-UseC_-Telco-Customer-Churn3.csv')

###############################################
########## Data Exploration ###################
# Print dataset statistical df
# Can check out outliers in numerical features
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
# Remove outlier

# Remove Numeric outlier
# Value below 0 because only positive values exist
df = df[df['MonthlyCharges'] > 0]
df = df[df['tenure'] > 0]

# show boxplot
def boxplot(col):
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df[[col]], color='red')
    # plt.show()

# show boxplot
# boxplot('tenure')
# boxplot('MonthlyCharges')

#Remove outlier
def outliers_iqr(data,col):
    q1, q3 = np.percentile(data[col],[25,75])
    iqr = q3-q1
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)

    outlier_idx = data[col][(data[col] < lower_bound) | (data[col] > upper_bound)].index
    return outlier_idx

tenure_outlier_index = outliers_iqr(df, 'tenure')
Charges_outlier_index = outliers_iqr(df,'MonthlyCharges')


print(df.loc[tenure_outlier_index, 'tenure'])
print(df.loc[Charges_outlier_index, 'MonthlyCharges'])

print(df.shape)
df = df.drop(tenure_outlier_index, axis=0)
# print(df.shape)

# Drop the rows with missing values.
df = df.dropna()

##################################################
# Only need to store actual meaningful values
# All other values changed NaN
# If you just use encoding, even strange values will be encoded.
# First of all, hit the Categorical Outlier

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
# # Remove Nan
df =  df.dropna()
print(df.info())

#########################################################################
#### Scaling feature
features_scaling = ['tenure', 'MonthlyCharges']
df_features_scaling = pd.DataFrame(df, columns = features_scaling)
df_remaining_features = df.drop(columns=features_scaling)

###### MaxAbsScaler

maxabs_scaler = preprocessing.MaxAbsScaler()
maxabs_features = maxabs_scaler.fit_transform(df_features_scaling)

# to change DataFrame
df_maxabs_features = \
    pd.DataFrame(maxabs_features, columns = features_scaling, index = df_remaining_features.index)

df_maxabs = pd.concat([df_remaining_features, df_maxabs_features], axis=1)                                           
print("MaxAbsScaler")
print(df_maxabs.head(),'\n')

################################################################################
# Show correlation plot for correlation of Churn with each of the remaining features
# maxabs correlation
df_maxabs.corr()['Churn'].sort_values(ascending=False).plot(kind='bar',figsize=(20,5))
plt.show()

# Split train and test data
X_maxabs1 = df_maxabs.drop('Churn', axis=1)
X_maxabs = X_maxabs1.values
y_maxabs=df_maxabs['Churn']

##################################################################################
maxabs_X_train, maxabs_X_test, maxabs_y_train, maxabs_y_test = train_test_split(X_maxabs,y_maxabs, test_size = 0.2, shuffle=False)

# Step Model Evaluation Metrics
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, classification_report,roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, plot_confusion_matrix, precision_score, recall_score

# Define a function that plots the feature weights for a classifier
def feature_weights(X_df, classifier, classifier_name):
    weights = pd.Series(classifier.coef_[0], index=X_df.columns.values).sort_values(ascending=False)

    top_weights_selected = weights[:10]
    plt.figure(figsize=(7,6))
    plt.tick_params(labelsize=10)
    plt.title(f'{classifier_name} - Top 10 Features')
    top_weights_selected.plot(kind = "bar")

    bottom_weights_selected = weights[-10:]
    plt.figure(figsize=(7,6))
    plt.tick_params(labelsize=10)
    plt.title(f'{classifier_name} - Bottom 10 Features')
    bottom_weights_selected.plot(kind = "bar")

    return print("")

# define a function that plots the confusion matrix
def confusion_matrix_plot(X_train, y_train, X_test, y_test, classifier, y_pred, classifier_name):
    fig, ax = plt.subplots(figsize=(7,6))
    #To plot confusiosn matrix
    plot_confusion_matrix(classifier, X_test, y_test, display_labels=["No Churn", "Churn"],
    cmap= plt.cm.Blues, normalize=None, ax=ax)
    ax.set_title(f'{classifier_name} - Confusion Matrix')
    plt.show()
    #To plot confusiosn matrix - norm
    fig, ax = plt.subplots(figsize=(7,6))
    plot_confusion_matrix(classifier, X_test, y_test, display_labels=["No Churn", "Churn"], 
    cmap= plt.cm.Blues, normalize='true', ax=ax)
    ax.set_title(f'{classifier_name} - Confusion Matrix (norm.)')
    plt.show()

    print(f'Accuracy Score Test: {accuracy_score(y_test,y_pred)}')
    print(f'Accuracy Score Train: {classifier.score(X_train, y_train)} (as comparison)')
    return print("")

# define a function that plots roc curve
def roc_curve_auc_score(X_test, y_test, y_pred_probabilities, classifier_name):
    # plot the roc curve for the model
    y_pred_prob = y_pred_probabilities[:,1]
    fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)
    # To show plot
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr,tpr,label = f'{classifier_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier_name} - ROC Curve')
    plt.show()

    return print(f'AUC Score (ROC) : {roc_auc_score(y_test, y_pred_prob)}\n')

# define a function that plots the precision-recall-curve
def precision_recall_curve_and_scores(X_test, y_test, y_pred, y_pred_probabilities, classifier_name):
    #To predict precision curve and scores
    y_pred_prob = y_pred_probabilities[:,1]
    precision,recall,thresholds = precision_recall_curve(y_test, y_pred_prob)
    #To show plot
    plt.plot(recall, precision, label=f'{classifier_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{classifier_name} - Precision-Recall Curve')
    plt.show()

    f1_score_result, auc_score_result = f1_score(y_test, y_pred), auc(recall, precision)

    return print(f'F1 Score : {f1_score_result} \nAuc Score (PR) : {auc_score_result}\n')

#########################################################
# KNN classifier
# Instanciate and train the logistic regression model based on the traning set
# MaxAbs
knn = KNeighborsClassifier()
knn.fit(maxabs_X_train, maxabs_y_train)

# make predictions
y_pred_knn = knn.predict(maxabs_X_test)
y_pred_knn_prob = knn.predict_proba(maxabs_X_test)

# Plot model evaluations
confusion_matrix_plot(maxabs_X_train,maxabs_y_train,maxabs_X_test,maxabs_y_test,knn,y_pred_knn, 'KNN')
roc_curve_auc_score(maxabs_X_test, maxabs_y_test, y_pred_knn_prob, 'KNN')
precision_recall_curve_and_scores(maxabs_X_test, maxabs_y_test, y_pred_knn, y_pred_knn_prob, 'KNN')

# Logistic Rrgression
# Instanciate and train the logistic regression model based on the traning set
logreg = LogisticRegression(max_iter=1000)
logreg.fit(maxabs_X_train, maxabs_y_train)

# make predictions
y_pred_logreg = logreg.predict(maxabs_X_test)
y_pred_logreg_prob = logreg.predict_proba(maxabs_X_test)

# Plot model evaluations
feature_weights(X_maxabs1, logreg, 'Log. Regression')
confusion_matrix_plot(maxabs_X_train, maxabs_y_train, maxabs_X_test, maxabs_y_test, logreg, y_pred_logreg, 'Log. Regression')
roc_curve_auc_score(maxabs_X_test, maxabs_y_test, y_pred_logreg_prob, 'Log. Regression')
precision_recall_curve_and_scores(maxabs_X_test, maxabs_y_test, y_pred_logreg, y_pred_logreg_prob, 'Log. Regression')

# Random Forest

# Instanciate and train the random forest model based on the training set

rf = RandomForestClassifier(n_jobs=-1, random_state=1)
rf.fit(maxabs_X_train, maxabs_y_train)

# make predictions

y_pred_rf = rf.predict(maxabs_X_test)
y_pred_rf_prob = rf.predict_proba(maxabs_X_test)

# Plot model evaluations

confusion_matrix_plot(maxabs_X_train, maxabs_y_train, maxabs_X_test, maxabs_y_test, rf, y_pred_rf, "Random Forest")
roc_curve_auc_score(maxabs_X_test, maxabs_y_test, y_pred_rf_prob, "Random Forest")
precision_recall_curve_and_scores(maxabs_X_test, maxabs_y_test, y_pred_rf, y_pred_rf_prob, "Random Forest")

# Random Forest's variable importance (which feature's characteristics are most important)
target = df.loc[:, ['Churn']]
rdata = df.drop(['Churn'], axis=1)
trainx, testx, trainy, testy = train_test_split(rdata, target, random_state=42)

# To define feature important values
ftr_importances_values = rf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = trainx.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(10,6))
plt.title('Top 20 Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()

