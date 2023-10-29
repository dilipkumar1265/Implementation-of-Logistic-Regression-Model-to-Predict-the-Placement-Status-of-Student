# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset. 4.Import LogisticRegression from sklearn and apply the model on the dataset.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Apply new unknown values.

## Program:
```
/*
Developed by: DILIP KUMAR R
RegisterNumber:  212222040037
*/
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(20)
dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)
dataset

dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset

dataset.shape
(215, 10)

dataset.info()

#catgorising for further labeling
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset.dtypes

dataset.info()

dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset
dataset.info()
dataset

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()
x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])
```

## Output:

![Screenshot 2023-10-02 193249](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/07846cf1-579e-4219-bf1a-21bbf37206b2)
![Screenshot 2023-10-02 193259](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/69009dc2-f6ff-4055-aab3-6f9450c10693)
![Screenshot 2023-10-02 193317](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/28943337-a6ec-41f9-afec-4dad41036e50)
![Screenshot 2023-10-02 193334](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/04a18050-7430-4917-b82d-157e935670c2)
![Screenshot 2023-10-02 193342](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/0cddde8a-1356-4613-8d21-c63360fa498a)
![Screenshot 2023-10-02 193353](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/4d68ec5f-b3e7-490a-8845-877d952c88a6)
![Screenshot 2023-10-02 193405](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/6d285129-0b82-470f-9b57-ba676f2c5b44)
![Screenshot 2023-10-02 193411](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/87b9d14b-71e7-44ad-97e6-333bdaef85d1)
![Screenshot 2023-10-02 193428](https://github.com/dilipkumar1265/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119065291/bae825b4-8d51-4e4d-aed3-220aa861ed31)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
