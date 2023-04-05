# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## Aim:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries such as pandas module to read the corresponding csv file.
2. Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the corresponding dataset values.
4. Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y and Predict the values of array using the variable y_pred.
5. Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
6. Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Yuvadarshini S
Register Number:  212221230126

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()


data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or column
data1.head()


data1.isnull().sum()


data1.duplicated().sum()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x


y=data1["status"]
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 0)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")# a library for large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
*/
```

## Output:
![41](https://user-images.githubusercontent.com/93482485/230104007-bfc6a3cd-28e6-4c46-bbe6-535147b4238b.jpg)
![42](https://user-images.githubusercontent.com/93482485/230104035-25bd27c7-4dbb-41ce-8339-4d88f02acf15.jpg)
![43](https://user-images.githubusercontent.com/93482485/230104071-d076bd1a-541e-49e6-aa2d-847d68be9055.jpg)
![44](https://user-images.githubusercontent.com/93482485/230104101-5c4107dd-9837-494d-a539-aa6295bbf871.jpg)
![45](https://user-images.githubusercontent.com/93482485/230104156-a5e4fad9-1581-4582-9fc5-071b10052948.jpg)
![46](https://user-images.githubusercontent.com/93482485/230104192-51a65e3d-e6a1-4313-93af-0671ac7dd5df.jpg)
![47](https://user-images.githubusercontent.com/93482485/230104250-24c455c3-f88a-4a49-88ee-ccefb45ac6b8.jpg)
![48](https://user-images.githubusercontent.com/93482485/230104296-9e040fb1-8a90-4257-87b9-12a19d26aa40.jpg)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
