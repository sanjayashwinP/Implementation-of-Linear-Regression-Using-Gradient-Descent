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
```

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

## Output:
![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/36f9a69f-f339-430a-b1aa-6e5472fd6699)

![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/815d4160-1a96-4ee8-9a5b-f4096880ae90)

![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/2f8f4e23-6cc9-434c-9753-d0b40d862f20) 

![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/b7f72898-1a9b-4a88-9836-b36fbf363e60)

![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/43c1de1d-955d-4a40-a852-2863fb2c1fad)

![image](https://github.com/sanjayashwinP/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147473265/8a0eb588-5ba3-48b6-8236-d7a69318173e)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
