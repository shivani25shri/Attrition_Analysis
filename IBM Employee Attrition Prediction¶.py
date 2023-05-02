#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


# In[22]:


employee_data = pd.read_csv("C:\\Users\\shivani shrivastava\\Downloads\\1576148666_ibmattritiondata.zip")
employee_data.head()


# # Data Cleaning & Preprocessing

# In[23]:


employee_data.isnull().sum()


# # Exploratory Data Analysis

# In[24]:


employee_data.shape


# In[25]:


employee_data.info()


# In[26]:


employee_data.select_dtypes(include=['object']).dtypes


# In[27]:


employee_data['Attrition'].value_counts()


# In[28]:


# let's encode the attrition column so we can use it for EDA
employee_data['Attrition'] = employee_data['Attrition'].factorize(['No','Yes'])[0]
employee_data.head()


# In[29]:


plt.figure(figsize=(8,8))
pie = employee_data.groupby('Attrition')['Attrition'].count()
plt.pie(pie, explode=[0.1, 0.1], labels=['No', 'Yes'], autopct='%1.1f%%');


# In[30]:


employee_data.select_dtypes(include=['int64']).dtypes


# ## Distribution of "Age"

# In[31]:


sns.distplot(employee_data["Age"])


# In[32]:


employee_data[['Age']].value_counts().sort_values(ascending=False).head(10)


# In[33]:


employee_data[['Age']].value_counts().sort_values(ascending=False).tail()


# ## Most employees are in their 30s with 35 year olds having the highest count and lowest are people at around the age 60 or less than 20.

# ## Plotting a Heatmap to assess correlations between different features¶

# In[34]:


sns.boxplot(employee_data["YearsAtCompany"])


# In[36]:


plt.figure(figsize=(8,6))
sns.countplot(x='Department', hue='Attrition', data=employee_data);


# In[37]:


employee_data['Department'].value_counts()


# Most attritions are from the research & development department only for sales department to come second by a small margin. HUman resources has the least number of attritions. But we need to keep in mind that R&D has a lot more employees than sales and HR.
# 
# If we considered percentage of attritions per department, we would see that the HR department has most attritions.

# In[41]:


sns.countplot(x='EducationField', hue='Attrition', data=employee_data);
plt.xticks(rotation=45)


# ## the degrees of employees really matter here as most of the number of attritions are similar.

# In[43]:


sns.countplot(x='EnvironmentSatisfaction', data=employee_data);


# ## Most employees seem to be satisfied with the working environment.

# ## Splitting Data

# In[45]:


# Separating the features from the target (In the process, we will drop features that we don't think are key factors.)
X = employee_data.drop(['Attrition','EducationField'],axis=1) # Features
y = employee_data['Attrition'] # Target


# In[46]:


# Label encoding the categorical variables

X['Department'] = preprocessing.LabelEncoder().fit_transform(X['Department'])
X['Education'] = preprocessing.LabelEncoder().fit_transform(X['Education'])
X['MaritalStatus'] = preprocessing.LabelEncoder().fit_transform(X['MaritalStatus'])


# In[47]:


# Data Standardization
Scaler = StandardScaler()
X = Scaler.fit_transform(X)


# In[48]:


# Splitting Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)


# In[49]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# # Model Building

# In[59]:


classifiers=[]


# In[60]:


model2 = LogisticRegression() 
classifiers.append(model2)


# In[61]:


classifiers


# In[65]:


for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clrep=classification_report(y_test,y_pred)
    print("\n\nAccuracy of %s is %s"%(clf, acc))
    print("\n\nClassification Report: \n%s"%(clrep))
    cm = confusion_matrix(ytest, ypred)
    print("Confusion Matrix of %s is\n %s"
          %(clf, cm))
    print("--------------------------------------------------------------------------")


# In[ ]:





# In[ ]:





# In[ ]:




