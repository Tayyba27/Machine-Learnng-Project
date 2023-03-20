# About Dataset
###According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
###This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

# Loading Libraries
import pandas as pd # Data Processing
import numpy as np # Array Processing
import os # Data Importing

# **DATA ANALYSIS**

import matplotlib.pyplot as plt # Plots 
import seaborn as sns # Graphs

# **PRE PROCESSING**

from sklearn.preprocessing import FunctionTransformer  # Transforming of Data
from sklearn.preprocessing import OneHotEncoder # Data Encoding
from sklearn.preprocessing import StandardScaler # Data Scaling
from imblearn.over_sampling import RandomOverSampler # Data OverSampling
from sklearn.decomposition import PCA # Principal Component Analysis

# **MODELS** 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# **NERURAL NETWORKS**

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# **METRICS**

from sklearn.metrics import accuracy_score # Model Classification Report

# Reading Data
import pandas as pd # Data Processing
import numpy as np
stroke_data = pd.read_csv("E:\PAID PROJECTS PAID PROJECTS\Stroke Prediction Dataset/healthcare-dataset-stroke-data.csv")
stroke_data.head()

type(stroke_data)

# Exploring Data
stroke_data.sample(3)

stroke_data.dtypes
stroke_data.info()
stroke_data.describe()
stroke_data.ndim
2
stroke_data.columns
Index(['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke'],
      dtype='object')
stroke_data["stroke"].nunique()

stroke_data.stroke.nunique()

stroke_data.stroke.unique()

stroke_data["stroke"].unique()
stroke_data["gender"].unique()

stroke_data.head(3)

stroke_data.stroke.value_counts(True)
stroke_data.stroke.value_counts().rename('count'),

stroke_data.stroke.value_counts(True).rename('%').mul(100)

stroke_data["stroke"].value_counts()

stroke_data["stroke"].sample(20)

stroke_data.stroke.value_counts()


# Show Number of Patient by Stroke 
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data=stroke_data , x='stroke')
plt.title('Number of Patient')

# features name
 
stroke_data.columns

stroke_data["stroke"].value_counts()

# Missing Values
stroke_data.isnull()


stroke_data.isnull().sum()

print('Missing data sum :')
print(stroke_data.isnull().sum())

print('\nMissing data percentage (%):')
print(stroke_data.isnull().sum()/stroke_data.count()*100)


# Seperate Categorical and Numerical Features
 
cat_features = [feature for feature in stroke_data.columns if stroke_data[feature].dtypes == 'O']
print('Number of categorical variables: ', len(cat_features))
print('*'*80)
print('Categorical variables column name:',cat_features)

cd = pd.DataFrame(cat_features)
cd.head()

numerical_features = [feature for feature in stroke_data.columns if stroke_data[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
print('*'*80)
print('Numerical Variables Column: ',numerical_features)

print('*'*10)


# Checking Duplicating Values
stroke_data.gender.duplicated()
stroke_data.duplicated().sum()

stroke_data['gender'].unique()

stroke_data['age'].nunique()

stroke_data['age'].sample(10)

stroke_data['hypertension'].unique()

stroke_data['heart_disease'].unique()

stroke_data['ever_married'].unique()

stroke_data['work_type'].unique()

stroke_data['Residence_type'].unique()

stroke_data['avg_glucose_level'].nunique()

stroke_data.columns

stroke_data['smoking_status'].unique()

stroke_data['stroke'].nunique()

stroke_data['stroke'].unique()


# Correlation matrix
corr = stroke_data.corr() 
plt.figure(figsize=(8,8))
sns.heatmap(data=corr, annot=True, cmap='Spectral').set(title="Correlation Matrix")


fig = plt.figure(figsize=(12,8))
corr = stroke_data.corr()
sns.heatmap(corr, linewidths=.5, cmap="RdBu", annot=True, fmt="g")

corr_matrix = stroke_data.corr().round(2)
corr_matrix  


mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, center=0, vmin=-1, vmax=1, mask=mask, annot=True, cmap='BrBG')
plt.show()

cat_features = [feature for feature in stroke_data.columns if stroke_data[feature].dtypes == 'O']
print('Number of categorical variables: ', len(cat_features))
print('*'*80)
print('Categorical variables column name:',cat_features)

numerical_features = [feature for feature in stroke_data.columns if stroke_data[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
print('*'*80)
print('Numerical Variables Column: ',numerical_features)

for col in cat_features[:]:
    plt.figure(figsize=(6,3), dpi=100)
    sns.countplot(data=stroke_data,x=col,hue ='stroke',palette='gist_rainbow_r')
    plt.legend(loc=(1.05,0.5))


# Barplot of numerical features:
#Plotting the barplot of numerical features
for col in numerical_features:
    plt.figure(figsize=(6,3), dpi=100)
    sns.barplot(data=stroke_data,x='stroke',y=col,palette='gist_rainbow_r')

    # Handling Missing Values
stroke_data.head()



stroke_data["bmi"]=stroke_data["bmi"].fillna(stroke_data["bmi"].mean())
stroke_data.isnull().sum()


# Dropping Irrelevent Columns

train  = stroke_data.drop(['id'],axis=1)
train

train_data_cat = train.select_dtypes("object")
train_data_num = train.select_dtypes("number")
train_data_cat.head(3)


train_data_cata_encoded=pd.get_dummies(train_data_cat, columns=train_data_cat.columns.to_list())
train_data_cata_encoded.head()

data=pd.concat([train_data_cata_encoded,train_data_num],axis=1,join="outer")
data.head()

# seperate dependant and independant feature
y = data['stroke']
x = data.drop('stroke', axis = 1)
print(x.shape)
print(y.shape)


# Scailing the data
sc = StandardScaler()
x = sc.fit_transform(x)
x


Splitting data into Training and Testing
#Importing our ML toolkit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.svm import SVC
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# Splitting the dataset
#training data 70%
#testing data 30%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
X_train.shape, X_test.shape


# Building Classifiers
 
accuracy = {}
# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred1 = lr.predict(X_test)
print(accuracy_score(y_test, y_pred1))
accuracy[str(lr)] = accuracy_score(y_test, y_pred1)*100

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred1)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

# Classification Report
print(classification_report(y_test,y_pred1))



# Predicting
y_pred_test = lr.predict(X_test)

test = pd.DataFrame({
    'Actual':y_test,
    'Y test predicted':y_pred_test
})
test.sample(10)

# DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3)
dtc.fit(X_train, y_train)
y_pred2 = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred2))
accuracy[str(dtc)] = accuracy_score(y_test, y_pred2)*100

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred2)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


print(classification_report(y_test,y_pred2))


y_pred_test = dtc.predict(X_test)

test = pd.DataFrame({
    'Actual':y_test,
    'Y test predicted':y_pred_test
})
test.head(5)

rfc = RandomForestClassifier(max_depth=5)
rfc.fit(X_train, y_train)
y_pred3 = rfc.predict(X_test)
print(accuracy_score(y_test, y_pred3))
accuracy[str(rfc)] = accuracy_score(y_test, y_pred3)*100

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred3)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# SVM
svc = SVC()
svc.fit(X_train, y_train)
y_pred5 = svc.predict(X_test)
print(accuracy_score(y_test, y_pred5))
accuracy[str(svc)] = accuracy_score(y_test, y_pred5)*100

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred5)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


#Conclusion
#Almost all heart disease people are above 50, which is obvious.
#Hypertesion disease in people of above 50.
#The most important features are "age", "bmi" and "glucose_level".
#Only 249 of the total dataset is positive for stroke(4.8%)
#This happens when we have very less(<5%) diagnosed positive for stroke i.e unbalanced target variable.
#We can use SMOTE(Synthetic Minority Oversampling Technique) to increase(oversample) the target varaible. It works by duplicating examples in the minority class.
 
# Handling this data using SMOTE
 
from imblearn.over_sampling import SMOTE

rom imblearn.over_sampling import SMOTE
smote = SMOTE()

x1, y1 = smote.fit_resample(x, y)

print(y_oversample.value_counts())
Splitting the oversampling data
X_train, X_test, y_train, y_test = train_test_split(x1,y1, test_size=0.3 ,shuffle = 'True',random_state = 3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred1 = lr.predict(X_test)
print(accuracy_score(y_test, y_pred1))
accuracy[str(lr)] = accuracy_score(y_test, y_pred1)*100
0.7919094960575934
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred1)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred1)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

print(classification_report(y_test,knn_predict))


y_pred_test = knn_model.predict(X_test)

test = pd.DataFrame({
    'Actual':y_test,
    'Y test predicted':y_pred_test
})


test.sample(10)








































































































































































































































































































































































