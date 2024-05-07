# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Intialize weights randomly.
2. Compute predicted.
3. Compute gradient of loss function.
4. Update weights using gradient descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SANJAY ASHWIN P
RegisterNumber: 212223040181
*/


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradiant
        theta=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    data=pd.read_csv("50_Startups.csv")
    data.head()   
    #Assuming the Lost column is your target variable 'y'd
X=(data.iloc [1:,:-2].values) 
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(- 1, 1) 
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:
![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/aed9fe7d-8472-4030-a629-8026b0536d48)


![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/c53f37c4-f3fa-4c94-9a8e-f5b77fb6829b)


![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/bb64e7aa-1a1c-4747-91b2-c5a045cdae23)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
