# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from seaborn.categorical import countplot
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

###################################################################

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

df = df.drop('customerID', axis=1)

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

###### Standard Scaler
standard_scaler = preprocessing.StandardScaler()
standard_features = standard_scaler.fit_transform(df_features_scaling)

# to change DataFrame
df_standard_features = \
    pd.DataFrame(standard_features, columns = features_scaling, index = df_remaining_features.index)

df_standard = pd.concat([df_remaining_features, df_standard_features], axis=1)                                           
print("MaxAbsScaler")
print(df_standard.head(),'\n')


################################################################################
# Show correlation plot for correlation of Churn with each of the remaining features
# Standard correlation
df_standard.corr()['Churn'].sort_values(ascending=False).plot(kind='bar',figsize=(20,5))
plt.show()

###################################
##### Split train and test data ###
###################################
X_standard1 = df_standard.drop('Churn', axis=1)
X_standard = X_standard1.values
y_standard=df_standard['Churn']
##################################################################################
standard_X_train, standard_X_test, standard_y_train, standard_y_test = \
    train_test_split(X_standard,y_standard, test_size = 0.2, shuffle=True)

# https://towardsdatascience.com/machine-learning-case-study-telco-customer-churn-prediction-bc4be03c9e1d
# Step Model Evaluation Metrics
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, classification_report,roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, plot_confusion_matrix, precision_score, recall_score

# DEfine a function that plots the feature weights for a classifier
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
    plot_confusion_matrix(classifier, X_test, y_test, display_labels=["No Churn", "Churn"], 
    cmap= plt.cm.Blues, normalize=None, ax=ax)
    ax.set_title(f'{classifier_name} - Confusion Matrix')
    plt.show()

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

    y_pred_prob = y_pred_probabilities[:,1]
    fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)

    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr,tpr,label = f'{classifier_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier_name} - ROC Curve')
    plt.show()

    return print(f'AUC Score (ROC) : {roc_auc_score(y_test, y_pred_prob)}\n')

# define a function that plots the precision-recall-curve
def precision_recall_curve_and_scores(X_test, y_test, y_pred, y_pred_probabilities, classifier_name):
    y_pred_prob = y_pred_probabilities[:,1]
    precision,recall,thresholds = precision_recall_curve(y_test, y_pred_prob)

    # To show plot
    plt.plot(recall, precision, label=f'{classifier_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{classifier_name} - Precision-Recall Curve')
    plt.show()

    f1_score_result, auc_score_result = f1_score(y_test, y_pred), auc(recall, precision)

    return print(f'F1 Score : {f1_score_result} \nAuc Score (PR) : {auc_score_result}\n')

######################################
############ KNN classifier ##########
######################################
# # Instanciate and train the logistic regression model based on the traning set
# MinMax
knn = KNeighborsClassifier()

# Standard
knn.fit(standard_X_train, standard_y_train)

# make predictions
y_pred_knn = knn.predict(standard_X_test)
y_pred_knn_prob = knn.predict_proba(standard_X_test)

# Plot model evaluations
confusion_matrix_plot(standard_X_train,standard_y_train,standard_X_test,standard_y_test,knn,y_pred_knn, 'KNN')
roc_curve_auc_score(standard_X_test, standard_y_test, y_pred_knn_prob, 'KNN')
precision_recall_curve_and_scores(standard_X_test, standard_y_test, y_pred_knn, y_pred_knn_prob, 'KNN')

##########################################
###### Logistic Rrgression ###############
##########################################
# Instanciate and train the logistic regression model based on the traning set
logreg = LogisticRegression(max_iter=1000)
logreg.fit(standard_X_train, standard_y_train)

# make predictions
y_pred_logreg = logreg.predict(standard_X_test)
y_pred_logreg_prob = logreg.predict_proba(standard_X_test)

# Plot model evaluations
feature_weights(X_standard1, logreg, 'Log. Regression')
confusion_matrix_plot(standard_X_train, standard_y_train, standard_X_test, standard_y_test, logreg, y_pred_logreg, 'Log. Regression')
roc_curve_auc_score(standard_X_test, standard_y_test, y_pred_logreg_prob, 'Log. Regression')
precision_recall_curve_and_scores(standard_X_test, standard_y_test, y_pred_logreg, y_pred_logreg_prob, 'Log. Regression')

######################################
############ Random Forest ##########
######################################

# Instanciate and train the random forest model based on the training set
rf = RandomForestClassifier()
rf.fit(standard_X_train, standard_y_train)

# make predictions
y_pred_rf = rf.predict(standard_X_test)
y_pred_rf_prob = rf.predict_proba(standard_X_test)

# Plot model evaluations
confusion_matrix_plot(standard_X_train, standard_y_train, standard_X_test, standard_y_test, rf, y_pred_rf, "Random Forest")
roc_curve_auc_score(standard_X_test, standard_y_test, y_pred_rf_prob, "Random Forest")
precision_recall_curve_and_scores(standard_X_test, standard_y_test, y_pred_rf, y_pred_rf_prob, "Random Forest")

########################################################
####### Hyperparameter Tuning/Model Improvement #########
######################################################
# To address a potential bias stemming from the specific split of the data in the train-test-split part, 
# cross-validation is used during hyperparameter tuning with Grid Search and Randomized Search. 
# Cross validations splits the training data into in a specified amount of folds. 
# Result of cross-validation is k values for all metrics on the k-fold CV.

print("###### Standard Scaling ########")
from sklearn.model_selection import GridSearchCV
# Define parameter grid for GridSearch and instanciate and train model
param_grid = {'n_neighbors' : np.arange(1,30)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv = 5)
knn_cv.fit(standard_X_train, standard_y_train)

# Make predictions (classes and probabilities) with the trained model on the test set
y_pred_knn_tuned = knn_cv.predict(standard_X_test)
y_pred_knn_tuned_prob = knn_cv.predict_proba(standard_X_test)
print('KNN best number of neighbors: ', knn_cv.best_params_,'\n')

confusion_matrix_plot(standard_X_train, standard_y_train, standard_X_test, standard_y_test, knn_cv, y_pred_knn_tuned, 'KNN (tuned)')
roc_curve_auc_score(standard_X_test, standard_y_test,y_pred_knn_tuned_prob, 'KNN (tuned)')
precision_recall_curve_and_scores(standard_X_test, standard_y_test, y_pred_knn_tuned, y_pred_knn_tuned_prob, 'KNN (tuned)')

#Define parameter grid for GridSearch and instanciate and train model
param_grid_L1 = {'penalty' : ['l1', 'l2'], 'C' : np.arange(.1,5,.1)}
logreg_tuned = LogisticRegression(solver = 'saga', max_iter=1000)
logreg_tuned_gs = GridSearchCV(logreg_tuned, param_grid_L1, cv =5)
logreg_tuned_gs.fit(standard_X_train, standard_y_train)

# Make predictions (calsses and probabilities) with the trained models on the test set.
y_pred_logreg_tuned = logreg_tuned_gs.predict(standard_X_test)
y_pred_logreg_tuned_prob = logreg_tuned_gs.predict_proba(standard_X_test)

print('Logistic Regression - Best Parameters: ', logreg_tuned_gs.best_params_)

#Plot model evaluations
confusion_matrix_plot(standard_X_train, standard_y_train, standard_X_test, standard_y_test, logreg_tuned_gs, y_pred_logreg_tuned, 'Log. Regression (tuned)')
roc_curve_auc_score(standard_X_test, standard_y_test,y_pred_logreg_tuned_prob, 'Log. Regression (tuned)')
precision_recall_curve_and_scores(standard_X_test, standard_y_test, y_pred_logreg_tuned, y_pred_logreg_tuned_prob, 'Log. Regression (tuned)')

from sklearn.model_selection import RandomizedSearchCV

# Define parameter grid for RandomizedSerarch and instanciate and train model
param_grid_rf = {'n_estimators' : np.arange(10, 2000, 10),
                'max_features' : ['auto','sqrt'],
                'max_depth' : np.arange(10,200,10),
                'criterion' : ['gini', 'entropy'],
                'bootstrap' : [True, False]}

rf = RandomForestClassifier()
rf_random_grid = RandomizedSearchCV(estimator = rf, param_distributions=param_grid_rf, cv = 5, verbose = 0)
rf_random_grid.fit(standard_X_train, standard_y_train)

# Make predictions (classes and probabilities) with the trained model on the test set.
y_pred_rf_tuned = rf_random_grid.predict(standard_X_test)
y_pred_rf_tuned_prob = rf_random_grid.predict_proba(standard_X_test)

print('Random Forest - Best Parameters: ', rf_random_grid.best_params_)

# Plot model evaluations
confusion_matrix_plot(standard_X_train, standard_y_train, standard_X_test, standard_y_test, rf_random_grid, y_pred_rf_tuned, 'Random Forest (tuned)')
roc_curve_auc_score(standard_X_test, standard_y_test,y_pred_rf_tuned_prob, 'Random Forest (tuned)')
precision_recall_curve_and_scores(standard_X_test, standard_y_test, y_pred_rf_tuned, y_pred_rf_tuned_prob, 'Random Forest (tuned)')
