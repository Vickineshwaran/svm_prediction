import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
diabetes = pd.read_csv("/content/diabetes.csv")
diabetes.head()
diabetes.tail()
diabetes.shape
diabetes.isnull().sum()
diabetes.describe()
diabetes['Outcome'].value_counts()
diabetes.groupby("Outcome").mean()
x = diabetes.drop(columns='Outcome',axis = 1)
y = diabetes['Outcome']
print(x)
print(y)
std = StandardScaler()
std.fit(x)
StandardScaler = std.transform(x)
X = StandardScaler
Y = diabetes['Outcome']
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
SVM = svm.SVC(kernel='linear')
SVM.fit(X_train,Y_train)
trans = SVM.predict(X_train)
train_predict = accuracy_score(trans,Y_train)
print(train_predict)
test = SVM.predict(X_test)
test_predict = accuracy_score(test,Y_test)
print(test_predict)
import warnings
warnings.filterwarnings("ignore")
if prediction[0] == 0:
    print("The person is non-diabetic")
else:
    print("The person is diabetic")
