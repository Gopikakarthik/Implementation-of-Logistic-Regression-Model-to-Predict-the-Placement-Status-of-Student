## Implementation of Logistic Regression Model to Predict the Placement Status of student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Find the null and duplicate values.
3. Using logistic regression find the predicted values of accuracy , confusion matrices.
4. Display the results.
 
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GOPIKA K
RegisterNumber:212222040046  
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
1.Placement Data

![image](https://github.com/user-attachments/assets/ffc73a11-8cbf-4b2b-a765-41c8bce8545b)

2.Salary Data

![image](https://github.com/user-attachments/assets/43e24a1e-1222-475a-96cf-53e3eb1af67c)

3.Checking the null function()

![image](https://github.com/user-attachments/assets/30e5f2f2-8686-4f14-a9b4-c506e5d76ff7)

4.Data Duplicate

![image](https://github.com/user-attachments/assets/80a0b4b1-9a00-43be-a907-479a0f48a7fe)

5.Print Data

![image](https://github.com/user-attachments/assets/5f7f1781-0c65-4a83-a88e-52d6ee301daf)
![image](https://github.com/user-attachments/assets/76f3cf8c-7801-4b25-a136-54f74868b8ec)

6.Data Status

![image](https://github.com/user-attachments/assets/2c84c5c8-261c-417b-9dca-51276f5d92ca)

7.y_prediction array

![image](https://github.com/user-attachments/assets/8880d2c3-79f6-43c0-b87c-71e359ec5f63)

8.Accuracy value

![image](https://github.com/user-attachments/assets/c192e99c-97dc-4e72-a063-1431ce48de33)

9.Confusion matrix

![image](https://github.com/user-attachments/assets/0ff0854e-15f0-4704-a0bc-12dff8097002)

10.Classification Report

![image](https://github.com/user-attachments/assets/f22b8f82-9fd6-4aa4-96ce-c796ce3dea0d)

11.Prediction of LR

![image](https://github.com/user-attachments/assets/2ada9c26-5baa-4347-84cf-c3e75d4b158c)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
