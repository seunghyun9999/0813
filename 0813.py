import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

header=['preg','plas','pres','skin','test',
         'mass','pedi','age','class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv',names=header)
array =data.values
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)
(X_train,X_text,
 Y_train,Y_test)=train_test_split(rescaled_X,Y,test_size=0.2)

model =LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_text)
y_pred_binary = (y_pred>0.5).astype(int)

acc = accuracy_score(y_pred_binary,Y_test)
print(acc)

df_Y_test = pd.DataFrame(Y_test)
df_Y_pred_binary = pd.DataFrame(y_pred_binary)
# df_Y_test.to_csv('./data/y_test.csv')
# df_Y_pred_binary.to_csv('./data/y_pred.csv')

plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)),Y_test,color='blue',label='Actual Values',
            marker='o')
plt.scatter(range(len(y_pred_binary)),y_pred_binary,color='r'
            ,label='Predicted Values',marker='x')
plt.title('CoA')
plt.xlabel('Index')
plt.ylabel('class 0,1')
plt.legend()
plt.show()





