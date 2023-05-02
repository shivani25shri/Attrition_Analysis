```python
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

```


```python
employee_data = pd.read_csv("C:\\Users\\shivani shrivastava\\Downloads\\1576148666_ibmattritiondata.zip")
employee_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EnvironmentSatisfaction</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>NumCompaniesWorked</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>2</td>
      <td>4</td>
      <td>Single</td>
      <td>5993</td>
      <td>8</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>3</td>
      <td>2</td>
      <td>Married</td>
      <td>5130</td>
      <td>1</td>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>4</td>
      <td>3</td>
      <td>Single</td>
      <td>2090</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>4</td>
      <td>3</td>
      <td>Married</td>
      <td>2909</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>2</td>
      <td>Married</td>
      <td>3468</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# Data Cleaning & Preprocessing


```python
employee_data.isnull().sum()
```




    Age                        0
    Attrition                  0
    Department                 0
    DistanceFromHome           0
    Education                  0
    EducationField             0
    EnvironmentSatisfaction    0
    JobSatisfaction            0
    MaritalStatus              0
    MonthlyIncome              0
    NumCompaniesWorked         0
    WorkLifeBalance            0
    YearsAtCompany             0
    dtype: int64



# Exploratory Data Analysis


```python
employee_data.shape
```




    (1470, 13)




```python
employee_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 13 columns):
     #   Column                   Non-Null Count  Dtype 
    ---  ------                   --------------  ----- 
     0   Age                      1470 non-null   int64 
     1   Attrition                1470 non-null   object
     2   Department               1470 non-null   object
     3   DistanceFromHome         1470 non-null   int64 
     4   Education                1470 non-null   int64 
     5   EducationField           1470 non-null   object
     6   EnvironmentSatisfaction  1470 non-null   int64 
     7   JobSatisfaction          1470 non-null   int64 
     8   MaritalStatus            1470 non-null   object
     9   MonthlyIncome            1470 non-null   int64 
     10  NumCompaniesWorked       1470 non-null   int64 
     11  WorkLifeBalance          1470 non-null   int64 
     12  YearsAtCompany           1470 non-null   int64 
    dtypes: int64(9), object(4)
    memory usage: 149.4+ KB
    


```python
employee_data.select_dtypes(include=['object']).dtypes
```




    Attrition         object
    Department        object
    EducationField    object
    MaritalStatus     object
    dtype: object




```python
employee_data['Attrition'].value_counts()
```




    No     1233
    Yes     237
    Name: Attrition, dtype: int64




```python
# let's encode the attrition column so we can use it for EDA
employee_data['Attrition'] = employee_data['Attrition'].factorize(['No','Yes'])[0]
employee_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EnvironmentSatisfaction</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>NumCompaniesWorked</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>1</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>2</td>
      <td>4</td>
      <td>Single</td>
      <td>5993</td>
      <td>8</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>0</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>3</td>
      <td>2</td>
      <td>Married</td>
      <td>5130</td>
      <td>1</td>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>4</td>
      <td>3</td>
      <td>Single</td>
      <td>2090</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>0</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>4</td>
      <td>3</td>
      <td>Married</td>
      <td>2909</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>0</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>2</td>
      <td>Married</td>
      <td>3468</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8,8))
pie = employee_data.groupby('Attrition')['Attrition'].count()
plt.pie(pie, explode=[0.1, 0.1], labels=['No', 'Yes'], autopct='%1.1f%%');
```


    
![png](output_10_0.png)
    



```python
employee_data.select_dtypes(include=['int64']).dtypes
```




    Age                        int64
    Attrition                  int64
    DistanceFromHome           int64
    Education                  int64
    EnvironmentSatisfaction    int64
    JobSatisfaction            int64
    MonthlyIncome              int64
    NumCompaniesWorked         int64
    WorkLifeBalance            int64
    YearsAtCompany             int64
    dtype: object



## Distribution of "Age"


```python
sns.distplot(employee_data["Age"])
```

    C:\Users\shivani shrivastava\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:xlabel='Age', ylabel='Density'>




    
![png](output_13_2.png)
    



```python
employee_data[['Age']].value_counts().sort_values(ascending=False).head(10)
```




    Age
    35     78
    34     77
    36     69
    31     69
    29     68
    32     61
    30     60
    38     58
    33     58
    40     57
    dtype: int64




```python
employee_data[['Age']].value_counts().sort_values(ascending=False).tail()
```




    Age
    59     10
    19      9
    18      8
    60      5
    57      4
    dtype: int64



## Most employees are in their 30s with 35 year olds having the highest count and lowest are people at around the age 60 or less than 20.

## Plotting a Heatmap to assess correlations between different features¶


```python
sns.boxplot(employee_data["YearsAtCompany"])
```

    C:\Users\shivani shrivastava\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='YearsAtCompany'>




    
![png](output_18_2.png)
    



```python
plt.figure(figsize=(8,6))
sns.countplot(x='Department', hue='Attrition', data=employee_data);
```


    
![png](output_19_0.png)
    



```python
employee_data['Department'].value_counts()
```




    Research & Development    961
    Sales                     446
    Human Resources            63
    Name: Department, dtype: int64



Most attritions are from the research & development department only for sales department to come second by a small margin. HUman resources has the least number of attritions. But we need to keep in mind that R&D has a lot more employees than sales and HR.

If we considered percentage of attritions per department, we would see that the HR department has most attritions.


```python
sns.countplot(x='EducationField', hue='Attrition', data=employee_data);
plt.xticks(rotation=45)
```




    (array([0, 1, 2, 3, 4, 5]),
     [Text(0, 0, 'Life Sciences'),
      Text(1, 0, 'Other'),
      Text(2, 0, 'Medical'),
      Text(3, 0, 'Marketing'),
      Text(4, 0, 'Technical Degree'),
      Text(5, 0, 'Human Resources')])




    
![png](output_22_1.png)
    


## the degrees of employees really matter here as most of the number of attritions are similar.


```python
sns.countplot(x='EnvironmentSatisfaction', data=employee_data);
```


    
![png](output_24_0.png)
    


## Most employees seem to be satisfied with the working environment.

## Splitting Data


```python
# Separating the features from the target (In the process, we will drop features that we don't think are key factors.)
X = employee_data.drop(['Attrition','EducationField'],axis=1) # Features
y = employee_data['Attrition'] # Target
```


```python
# Label encoding the categorical variables

X['Department'] = preprocessing.LabelEncoder().fit_transform(X['Department'])
X['Education'] = preprocessing.LabelEncoder().fit_transform(X['Education'])
X['MaritalStatus'] = preprocessing.LabelEncoder().fit_transform(X['MaritalStatus'])
```


```python
# Data Standardization
Scaler = StandardScaler()
X = Scaler.fit_transform(X)
```


```python
# Splitting Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)
```


```python
print(X.shape)
print(X_train.shape)
print(X_test.shape)
```

    (1470, 11)
    (1176, 11)
    (294, 11)
    

# Model Building


```python
classifiers=[]
```


```python
model2 = LogisticRegression() 
classifiers.append(model2)
```


```python
classifiers

```




    [LogisticRegression()]




```python
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
```

    
    
    Accuracy of LogisticRegression() is 0.8469387755102041
    
    
    Classification Report: 
                  precision    recall  f1-score   support
    
               0       0.85      1.00      0.92       246
               1       0.80      0.08      0.15        48
    
        accuracy                           0.85       294
       macro avg       0.82      0.54      0.53       294
    weighted avg       0.84      0.85      0.79       294
    
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    C:\Users\SHIVAN~1\AppData\Local\Temp/ipykernel_17156/811685003.py in <module>
          6     print("\n\nAccuracy of %s is %s"%(clf, acc))
          7     print("\n\nClassification Report: \n%s"%(clrep))
    ----> 8     cm = confusion_matrix(ytest, ypred)
          9     print("Confusion Matrix of %s is\n %s"
         10           %(clf, cm))
    

    NameError: name 'confusion_matrix' is not defined



```python

```


```python

```


```python

```
